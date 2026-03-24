# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure Vulkan Model Weight Loader — loads safetensors directly into Vulkan memory.

This bypasses PyTorch CUDA entirely for weight loading. The flow:
1. Parse safetensors file metadata (header JSON)
2. Memory-map the file on CPU
3. For each weight tensor:
   a. Allocate VulkanBuffer of the right size
   b. Upload weight data via Vulkan staging buffer
   c. Store metadata (name, shape, dtype, device_address)

This is used when VLLM_VULKAN_GEMM_STRATEGY=cutlass_ptx for 100%
nvidia_uvm elimination. With cuBLAS shim, PyTorch model loading is used.
"""

import json
import struct as struct_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

# safetensors dtype mapping
_ST_DTYPE_MAP = {
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (np.float16, 2),  # numpy doesn't have bf16, handle separately
    "I32": (np.int32, 4),
    "I64": (np.int64, 8),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


@dataclass
class VulkanWeight:
    """A model weight tensor stored in Vulkan GPU memory."""
    name: str
    shape: tuple[int, ...]
    dtype: str  # safetensors dtype string (F32, F16, BF16, etc.)
    device_address: int  # GPU virtual address from vkGetBufferDeviceAddress
    buffer: Any  # VulkanBuffer handle
    nbytes: int

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result


@dataclass
class VulkanModel:
    """A model loaded entirely in Vulkan GPU memory."""
    weights: dict[str, VulkanWeight] = field(default_factory=dict)
    total_bytes: int = 0

    def get_address(self, name: str) -> int:
        """Get the GPU device address for a named weight."""
        return self.weights[name].device_address

    def __repr__(self):
        return (f"VulkanModel({len(self.weights)} weights, "
                f"{self.total_bytes / (1024**3):.2f} GiB)")


def parse_safetensors_header(file_path: str | Path) -> dict:
    """Parse the JSON header from a safetensors file."""
    path = Path(file_path)
    with open(path, "rb") as f:
        # First 8 bytes: header size as uint64 LE
        header_size = struct_mod.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        return json.loads(header_bytes)


def load_safetensors_to_vulkan(
    file_path: str | Path,
    vulkan_ctx: Any,
    weight_filter: set[str] | None = None,
) -> VulkanModel:
    """
    Load a safetensors file directly into Vulkan GPU memory.

    Args:
        file_path: Path to .safetensors file
        vulkan_ctx: VulkanDeviceContext instance
        weight_filter: Optional set of weight names to load (None = all)

    Returns:
        VulkanModel with all weights in GPU memory
    """
    path = Path(file_path)
    logger.info("Loading %s into Vulkan GPU memory...", path.name)

    # Parse header
    header = parse_safetensors_header(path)

    # Remove metadata entry if present
    metadata = header.pop("__metadata__", {})

    model = VulkanModel()

    # Memory-map the file for efficient reading
    with open(path, "rb") as f:
        # Read header size
        header_size = struct_mod.unpack("<Q", f.read(8))[0]
        data_offset = 8 + header_size  # Start of tensor data

        for name, tensor_info in header.items():
            if weight_filter and name not in weight_filter:
                continue

            dtype_str = tensor_info["dtype"]
            shape = tuple(tensor_info["shape"])
            offsets = tensor_info["data_offsets"]  # [start, end] relative to data start

            if dtype_str not in _ST_DTYPE_MAP:
                logger.warning("Skipping %s: unsupported dtype %s",
                               name, dtype_str)
                continue

            np_dtype, elem_size = _ST_DTYPE_MAP[dtype_str]
            start = data_offset + offsets[0]
            end = data_offset + offsets[1]
            nbytes = end - start

            # Read weight data from file
            f.seek(start)
            weight_bytes = f.read(nbytes)

            # Allocate Vulkan GPU buffer
            buf = vulkan_ctx.allocate(nbytes)

            # Upload to GPU
            vulkan_ctx.upload(weight_bytes, buf)

            weight = VulkanWeight(
                name=name,
                shape=shape,
                dtype=dtype_str,
                device_address=buf.device_address,
                buffer=buf,
                nbytes=nbytes,
            )
            model.weights[name] = weight
            model.total_bytes += nbytes

            logger.debug("  Loaded %s: shape=%s dtype=%s addr=0x%x (%d bytes)",
                          name, shape, dtype_str, buf.device_address, nbytes)

    logger.info("Loaded %d weights (%s) into Vulkan GPU memory",
                 len(model.weights), _format_size(model.total_bytes))
    return model


def load_model_sharded(
    model_path: str | Path,
    vulkan_ctx: Any,
    weight_filter: set[str] | None = None,
) -> VulkanModel:
    """
    Load a sharded model (multiple safetensors files) into Vulkan memory.

    Handles the model.safetensors.index.json pattern used by HuggingFace.
    """
    model_dir = Path(model_path)

    # Check for index file (sharded model)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]  # name -> filename

        # Group weights by file
        files: dict[str, list[str]] = {}
        for weight_name, filename in weight_map.items():
            if weight_filter and weight_name not in weight_filter:
                continue
            files.setdefault(filename, []).append(weight_name)

        model = VulkanModel()
        for filename, names in files.items():
            file_path = model_dir / filename
            partial = load_safetensors_to_vulkan(
                file_path, vulkan_ctx, weight_filter=set(names))
            model.weights.update(partial.weights)
            model.total_bytes += partial.total_bytes

        return model

    # Single safetensors file
    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        return load_safetensors_to_vulkan(single_path, vulkan_ctx,
                                           weight_filter)

    # Try all .safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if st_files:
        model = VulkanModel()
        for f in st_files:
            partial = load_safetensors_to_vulkan(f, vulkan_ctx, weight_filter)
            model.weights.update(partial.weights)
            model.total_bytes += partial.total_bytes
        return model

    raise FileNotFoundError(
        f"No safetensors files found in {model_dir}. "
        f"Ensure the model is downloaded in safetensors format.")


def _format_size(nbytes: int) -> str:
    """Format bytes as human-readable string."""
    if nbytes >= 1024**3:
        return f"{nbytes / (1024**3):.2f} GiB"
    elif nbytes >= 1024**2:
        return f"{nbytes / (1024**2):.1f} MiB"
    else:
        return f"{nbytes / 1024:.0f} KiB"
