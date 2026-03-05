# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Tests for SglangRollout.release(tags=...) used in torch_memory_saver resync.

Verifies that:
1. release() defaults to ["kv_cache", "weights"] when no tags specified
2. release(tags=["weights"]) only releases weights
3. The resync pattern (release + resume) calls the engine correctly
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter


@pytest.fixture
def mock_adapter():
    """Create a ServerAdapter with mocked internals."""
    config = MagicMock()
    config.free_cache_engine = True
    config.tensor_parallel_size = 1
    config.checkpoint_engine.update_weights_bucket_megabytes = 64

    adapter = ServerAdapter.__new__(ServerAdapter)
    adapter.config = config
    adapter._engine = AsyncMock()
    adapter._engine.release_memory_occupation = AsyncMock(return_value={})
    adapter._engine.resume_memory_occupation = AsyncMock(return_value={})
    adapter._server_adapter_initialized = True

    # Mock device_mesh for rank check
    mock_mesh = MagicMock()
    mock_mesh.get_local_rank.return_value = 0
    adapter.device_mesh = {"infer_tp": mock_mesh}

    return adapter


@pytest.mark.asyncio
async def test_release_default_tags(mock_adapter):
    """release() without tags should release both kv_cache and weights."""
    await mock_adapter.release()
    mock_adapter._engine.release_memory_occupation.assert_awaited_once_with(tags=["kv_cache", "weights"])


@pytest.mark.asyncio
async def test_release_weights_only(mock_adapter):
    """release(tags=["weights"]) should only release weights."""
    await mock_adapter.release(tags=["weights"])
    mock_adapter._engine.release_memory_occupation.assert_awaited_once_with(tags=["weights"])


@pytest.mark.asyncio
async def test_release_kv_cache_only(mock_adapter):
    """release(tags=["kv_cache"]) should only release kv_cache."""
    await mock_adapter.release(tags=["kv_cache"])
    mock_adapter._engine.release_memory_occupation.assert_awaited_once_with(tags=["kv_cache"])


@pytest.mark.asyncio
async def test_memory_saver_resync_pattern(mock_adapter):
    """The resync pattern (release weights + resume weights) should call engine correctly."""
    # This is the pattern used in fsdp_workers.py after update_weights
    await mock_adapter.release(tags=["weights"])
    await mock_adapter.resume(tags=["weights"])

    mock_adapter._engine.release_memory_occupation.assert_awaited_once_with(tags=["weights"])
    mock_adapter._engine.resume_memory_occupation.assert_awaited_once_with(tags=["weights"])


@pytest.mark.asyncio
async def test_release_skipped_when_not_leader(mock_adapter):
    """release() should be a no-op when not the local leader rank."""
    mock_adapter.device_mesh["infer_tp"].get_local_rank.return_value = 1
    await mock_adapter.release(tags=["weights"])
    mock_adapter._engine.release_memory_occupation.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_skipped_when_free_cache_disabled(mock_adapter):
    """release() should be a no-op when free_cache_engine is False."""
    mock_adapter.config.free_cache_engine = False
    await mock_adapter.release(tags=["weights"])
    mock_adapter._engine.release_memory_occupation.assert_not_awaited()
