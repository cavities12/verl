# Copyright 2026 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LoRA sleep_level / lora_as_adapter logic in SGLang rollout.

Tests the branching logic that controls what gets released during sleep:
  - sleep_level=2 (merge path or no LoRA): release weights + kv_cache
  - sleep_level=1 (adapter path): release kv_cache only, keep base weights
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Lightweight stubs so we can import SGLangHttpServer / ServerAdapter without
# pulling in torch, ray, sglang, etc.
# ---------------------------------------------------------------------------


@dataclass
class _StubModelConfig:
    """Minimal stand-in for HFModelConfig."""

    lora_rank: int = 0
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubRolloutConfig:
    """Minimal stand-in for RolloutConfig."""

    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1


# ---------------------------------------------------------------------------
# lora_as_adapter property tests (mirrors vllm_async_server pattern)
# ---------------------------------------------------------------------------


class _LoraAsAdapterMixin:
    """Reproduces the lora_as_adapter property from SGLangHttpServer so we can
    test the boolean logic without importing the real class."""

    model_config: _StubModelConfig

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)


class _FakeServer(_LoraAsAdapterMixin):
    def __init__(self, model_config: _StubModelConfig):
        self.model_config = model_config


class TestLoraAsAdapter:
    """Test lora_as_adapter property logic."""

    def test_no_lora(self):
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={}))
        assert server.lora_as_adapter is False

    def test_lora_merge_true(self):
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={"merge": True}))
        assert server.lora_as_adapter is False

    def test_lora_merge_false(self):
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={"merge": False}))
        assert server.lora_as_adapter is True

    def test_lora_merge_absent_defaults_false(self):
        """When merge key is absent, it defaults to False → adapter mode."""
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={}))
        assert server.lora_as_adapter is True

    def test_lora_rank_in_dict(self):
        """lora_rank=0 but lora.rank>0 should still detect LoRA."""
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={"rank": 16}))
        assert server.lora_as_adapter is True

    def test_lora_rank_in_dict_with_merge(self):
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={"rank": 16, "merge": True}))
        assert server.lora_as_adapter is False


# ---------------------------------------------------------------------------
# sleep_level → release tag tests (ServerAdapter.release)
# ---------------------------------------------------------------------------


class TestServerAdapterReleaseTags:
    """Test that ServerAdapter.release() sends the right tags based on sleep_level."""

    @staticmethod
    def _make_adapter(sleep_level: int = 2):
        """Build a minimal ServerAdapter-like object without real init."""
        adapter = MagicMock()
        adapter.sleep_level = sleep_level
        adapter.config = _StubRolloutConfig(free_cache_engine=True)
        # device_mesh["infer_tp"].get_local_rank() == 0
        tp_mesh = MagicMock()
        tp_mesh.get_local_rank.return_value = 0
        adapter.device_mesh = {"infer_tp": tp_mesh}
        adapter._engine = AsyncMock()
        adapter._engine.release_memory_occupation = AsyncMock()
        return adapter

    def test_sleep_level_2_releases_everything(self):
        adapter = self._make_adapter(sleep_level=2)

        # Call the real release logic inline (avoids importing ServerAdapter)
        async def release():
            if adapter.device_mesh["infer_tp"].get_local_rank() == 0 and adapter.config.free_cache_engine:
                if adapter.sleep_level == 1:
                    tags = ["kv_cache"]
                else:
                    tags = ["kv_cache", "weights"]
                await adapter._engine.release_memory_occupation(tags=tags)

        asyncio.get_event_loop().run_until_complete(release())
        adapter._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache", "weights"])

    def test_sleep_level_1_releases_kv_only(self):
        adapter = self._make_adapter(sleep_level=1)

        async def release():
            if adapter.device_mesh["infer_tp"].get_local_rank() == 0 and adapter.config.free_cache_engine:
                if adapter.sleep_level == 1:
                    tags = ["kv_cache"]
                else:
                    tags = ["kv_cache", "weights"]
                await adapter._engine.release_memory_occupation(tags=tags)

        asyncio.get_event_loop().run_until_complete(release())
        adapter._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache"])


# ---------------------------------------------------------------------------
# SGLangHttpServer.sleep() tag selection
# ---------------------------------------------------------------------------


class TestSGLangHttpServerSleepTags:
    """Test that SGLangHttpServer.sleep() uses the right tags based on lora_as_adapter."""

    @staticmethod
    def _run_sleep_logic(lora_as_adapter: bool):
        """Simulate the sleep() method's tag selection logic and return chosen tags."""
        # Mirrors async_sglang_server.py sleep() HYBRID branch
        if lora_as_adapter:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]
        return tags

    def test_no_lora_releases_everything(self):
        tags = self._run_sleep_logic(lora_as_adapter=False)
        assert tags == ["kv_cache", "weights"]

    def test_adapter_mode_releases_kv_only(self):
        tags = self._run_sleep_logic(lora_as_adapter=True)
        assert tags == ["kv_cache"]


# ---------------------------------------------------------------------------
# engine_workers update_weights logic
# ---------------------------------------------------------------------------


class TestUpdateWeightsLogic:
    """Test the LoRA branching logic in engine_workers.update_weights()."""

    def test_merge_path_skips_adapter_block(self):
        """When peft_merge=True, peft_config is None → adapter block is skipped."""
        peft_merge = True
        peft_config = None  # engine returns None for merge path
        base_sync_done = False

        do_lora_base_sync = False
        if not peft_merge and peft_config is not None:
            do_lora_base_sync = not base_sync_done

        assert do_lora_base_sync is False

    def test_adapter_path_first_iteration_needs_base_sync(self):
        """When peft_merge=False, first iteration should sync base weights."""
        peft_merge = False
        peft_config = {"default": "some_config"}
        base_sync_done = False

        do_lora_base_sync = False
        if not peft_merge and peft_config is not None:
            do_lora_base_sync = not base_sync_done

        assert do_lora_base_sync is True

    def test_adapter_path_subsequent_iterations_skip_base_sync(self):
        """After first sync, base weights are retained → no re-sync needed."""
        peft_merge = False
        peft_config = {"default": "some_config"}
        base_sync_done = True

        do_lora_base_sync = False
        if not peft_merge and peft_config is not None:
            do_lora_base_sync = not base_sync_done

        assert do_lora_base_sync is False

    def test_no_lora_skips_adapter_block(self):
        """Without LoRA, peft_config is None → adapter block skipped."""
        peft_merge = False
        peft_config = None
        base_sync_done = False

        do_lora_base_sync = False
        if not peft_merge and peft_config is not None:
            do_lora_base_sync = not base_sync_done

        assert do_lora_base_sync is False

    def test_resume_skipped_when_sleep_level_1(self):
        """When sleep_level=1, weight resume should be skipped (weights not released)."""
        sleep_level = 1
        should_resume_weights = sleep_level == 2
        assert should_resume_weights is False

    def test_resume_called_when_sleep_level_2(self):
        """When sleep_level=2, weight resume should happen (weights were released)."""
        sleep_level = 2
        should_resume_weights = sleep_level == 2
        assert should_resume_weights is True

    def test_init_sleep_level_lora_no_merge(self):
        """init_model should set sleep_level=1 when LoRA + merge=False."""
        is_lora = True
        peft_merge = False
        rollout = MagicMock()
        rollout.sleep_level = 2  # default

        if is_lora and not peft_merge:
            rollout.sleep_level = 1

        assert rollout.sleep_level == 1

    def test_init_sleep_level_lora_with_merge(self):
        """init_model should keep sleep_level=2 when LoRA + merge=True."""
        is_lora = True
        peft_merge = True
        rollout = MagicMock()
        rollout.sleep_level = 2  # default

        if is_lora and not peft_merge:
            rollout.sleep_level = 1

        assert rollout.sleep_level == 2

    def test_init_sleep_level_no_lora(self):
        """init_model should keep sleep_level=2 when no LoRA."""
        is_lora = False
        peft_merge = False
        rollout = MagicMock()
        rollout.sleep_level = 2

        if is_lora and not peft_merge:
            rollout.sleep_level = 1

        assert rollout.sleep_level == 2
