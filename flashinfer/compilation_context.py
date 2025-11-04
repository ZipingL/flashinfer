"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Global compilation context management for FlashInfer.
"""

import os
import logging
import re
import torch

logger = logging.getLogger(__name__)


def _parse_arch_list(arch_list: str) -> set[tuple[int, str]]:
    """
    Parse a space separated list of CUDA architectures into (major, minor) tuples.
    Accepted formats:
        - "7.5", "8.0", "9.0a"
        - "sm_80", "compute_90a"
        - "120", "120a"
    """
    targets: set[tuple[int, str]] = set()
    if not arch_list:
        return targets
    for raw_token in arch_list.split():
        token = raw_token.strip().lower()
        if not token:
            continue
        if token.startswith("sm_"):
            token = token[3:]
        elif token.startswith("compute_"):
            token = token[8:]

        major_str: str
        minor_str: str
        if "." in token:
            major_str, minor_str = token.split(".", 1)
        elif re.fullmatch(r"\d+[a-z]", token):
            major_str, minor_str = token[:-1], token[-1:]
        elif re.fullmatch(r"\d+[a-z]{2}", token):
            major_str, minor_str = token[:-2], token[-2:]
        elif len(token) > 1 and token[:-1].isdigit() and token[-1].isdigit():
            major_str, minor_str = token[:-1], token[-1:]
        else:
            # Fallback: try to split digits / letters
            match = re.fullmatch(r"(\d+)([0-9a-z]*)", token)
            if not match:
                continue
            major_str = match.group(1)
            minor_str = match.group(2)

        if not major_str:
            continue

        try:
            major = int(major_str)
        except ValueError:
            continue

        if not minor_str:
            minor_str = "0"
        targets.add((major, minor_str))

    return targets


def _env_flag_is_true(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


class CompilationContext:
    COMMON_NVCC_FLAGS = [
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
    ]

    def __init__(self):
        self.TARGET_CUDA_ARCHS = set()
        if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
            self.TARGET_CUDA_ARCHS.update(
                _parse_arch_list(os.environ["FLASHINFER_CUDA_ARCH_LIST"])
            )
        else:
            try:
                for device in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(device)
                    if major >= 9:
                        minor = str(minor) + "a"
                    self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
            except Exception as e:
                logger.warning(f"Failed to get device capability: {e}.")

        extra_archs = os.environ.get("FLASHINFER_EXTRA_CUDA_ARCH_LIST")
        if extra_archs:
            self.TARGET_CUDA_ARCHS.update(_parse_arch_list(extra_archs))

        if _env_flag_is_true("FLASHINFER_ENABLE_SM120"):
            self.TARGET_CUDA_ARCHS.add((12, "0"))

        if _env_flag_is_true("FLASHINFER_ENABLE_SM120A"):
            self.TARGET_CUDA_ARCHS.add((12, "0a"))

    def get_nvcc_flags_list(
        self, supported_major_versions: list[int] = None
    ) -> list[str]:
        if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]
        else:
            supported_cuda_archs = self.TARGET_CUDA_ARCHS
        if len(supported_cuda_archs) == 0:
            raise RuntimeError(
                f"No supported CUDA architectures found for major versions {supported_major_versions}."
            )
        return [
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            for major, minor in supported_cuda_archs
        ] + self.COMMON_NVCC_FLAGS
