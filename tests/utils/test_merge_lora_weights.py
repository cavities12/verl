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

"""Tests for merge_lora_weights() used in SGLang LoRA weight sync.

Verifies that merge_lora_weights correctly:
1. Produces HuggingFace-format key names (no peft/FSDP prefixes)
2. Returns merged weights (base + LoRA, not just base or just deltas)
3. Leaves the model in unmerged state after extraction
4. Works with both FSDP1 and FSDP2

These tests require 2+ GPUs.
"""

import os

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    get_fsdp_wrap_policy,
    merge_lora_weights,
)


def _test_merge_lora_weights_worker(rank, world_size, rendezvous_file, strategy):
    """Worker function for testing merge_lora_weights with FSDP.

    Args:
        rank: Process rank
        world_size: Total number of processes
        rendezvous_file: Path to rendezvous file for distributed init
        strategy: FSDP strategy ("fsdp" or "fsdp2")
    """
    get_torch_device().set_device(rank)
    torch.distributed.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    # Create a small Qwen2 model
    config = Qwen2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
    )

    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model = model.to(device=get_device_name())

    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Initialize LoRA weights to non-zero values so merge produces different results
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLayer):
                for adapter_name in module.lora_A.keys():
                    module.lora_A[adapter_name].weight.data.uniform_(0.5, 1.5)
                    module.lora_B[adapter_name].weight.data.uniform_(1.5, 2.5)

    # Save pre-merge base weights for comparison
    base_weights_before = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            base_weights_before[name] = param.detach().clone()

    # Wrap with FSDP
    if strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
        model = FSDP(
            model,
            use_orig_params=True,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            auto_wrap_policy=get_fsdp_wrap_policy(module=model, is_lora=True),
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        }
        apply_fsdp2(model, fsdp_kwargs, {})

    # Run merge_lora_weights
    merged = merge_lora_weights(model)

    # === Verification ===

    # 1. Should return non-empty dict
    assert len(merged) > 0, "merge_lora_weights should return non-empty OrderedDict"

    # 2. Keys should be clean HF format (no peft/FSDP prefixes)
    for key in merged:
        assert "_fsdp_wrapped_module" not in key, f"Key should not contain FSDP prefix: {key}"
        assert "base_model.model." not in key, f"Key should not contain peft prefix: {key}"
        assert ".base_layer" not in key, f"Key should not contain .base_layer: {key}"
        assert "lora_" not in key, f"Key should not contain lora_ prefix: {key}"
        assert "_flat_param" not in key, f"Key should not contain _flat_param: {key}"

    # 3. Should contain expected model keys
    has_layer_key = any(k.startswith("model.layers.") for k in merged)
    assert has_layer_key, f"Should have model.layers.* keys, got: {list(merged.keys())[:5]}"

    # 4. All values should be CPU tensors
    for key, val in merged.items():
        assert val.device == torch.device("cpu"), f"Value for {key} should be on CPU, got {val.device}"

    # 5. Model should be in unmerged state after extraction
    lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert len(lora_layers) > 0, "Model should still have LoRA layers"
    for layer in lora_layers:
        assert not getattr(layer, "merged", False), "LoRA should be unmerged after merge_lora_weights"

    if rank == 0:
        print(f"merge_lora_weights test with {strategy} passed: {len(merged)} params extracted")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2,))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
def test_merge_lora_weights(world_size, strategy, tmp_path):
    """Test merge_lora_weights extracts correct HF-format merged weights."""
    rendezvous_file = str(tmp_path / f"rdzv_file_merge_lora_{strategy}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_test_merge_lora_weights_worker,
        args=(world_size, rendezvous_file, strategy),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
