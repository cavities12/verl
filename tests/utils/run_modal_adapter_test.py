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
"""Run adapter path integration test on Modal with 1 A100 GPU.

Tests the LoRA adapter path (lora.merge=False):
1. Launch sglang with a small model (Qwen2.5-0.5B-Instruct)
2. Create a LoRA adapter (peft)
3. Sync base weights (base_sync_done=False path)
4. Load adapter via LoadLoRAAdapterFromTensorsReqInput (base_sync_done=True path)
5. Generate and verify output is coherent (not garbage)

Usage:
    modal run tests/utils/run_modal_adapter_test.py
"""

from pathlib import Path

import modal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libnuma-dev")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate",
        "pytest",
        "safetensors",
        "omegaconf",
        "hydra-core",
        "psutil",
        "codetiming",
    )
    .run_commands(
        # Install sglang + flashinfer with matching versions
        "pip install 'sglang>=0.5.9' --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python",
    )
    .add_local_dir(
        str(_PROJECT_ROOT / "verl"),
        remote_path="/root/verl_src/verl",
        copy=True,
        ignore=["__pycache__", "*.pyc", "*.egg-info", "sglang"],
    )
    .add_local_file(
        str(_PROJECT_ROOT / "tests" / "utils" / "test_adapter_path_integration.py"),
        remote_path="/root/test_adapter_path_integration.py",
        copy=True,
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App("verl-adapter-test", image=image)


@app.function(gpu="A100", timeout=900)
def run_tests():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/verl_src:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/test_adapter_path_integration.py",
            "-v",
            "--tb=long",
            "-s",
        ],
        capture_output=True,
        text=True,
        timeout=800,
        env=env,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-12000:])
    return result.returncode


@app.local_entrypoint()
def main():
    rc = run_tests.remote()
    print(f"\nExit code: {rc}")
    if rc != 0:
        raise SystemExit(rc)
