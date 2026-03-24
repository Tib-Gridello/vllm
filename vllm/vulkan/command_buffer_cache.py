# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vulkan Command Buffer Cache — equivalent to CUDA Graphs.

CUDA Graphs pre-record a sequence of GPU operations and replay them with
minimal CPU overhead. In Vulkan, the equivalent is pre-recording command
buffers and resubmitting them.

Key differences from CUDA Graphs:
- Vulkan command buffers can be re-recorded (CUDA graphs are immutable)
- Vulkan submission has slightly more overhead than CUDA graph replay
- Vulkan doesn't have "graph capture" — we explicitly build the sequence

The approach:
1. On first inference pass with a given batch shape, record all kernel
   launches into a Vulkan command buffer
2. On subsequent passes with the same shape, replay the pre-recorded
   command buffer (skip all CPU-side dispatch logic)
3. If buffer addresses change (due to reallocation), re-record

This is primarily useful for the decode path where batch size and sequence
structure are relatively stable between steps.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class BatchDescriptor:
    """Describes a batch shape for command buffer caching."""

    def __init__(self, num_tokens: int, num_seqs: int,
                 max_seq_len: int, num_layers: int):
        self.num_tokens = num_tokens
        self.num_seqs = num_seqs
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

    def __hash__(self):
        return hash((self.num_tokens, self.num_seqs,
                     self.max_seq_len, self.num_layers))

    def __eq__(self, other):
        if not isinstance(other, BatchDescriptor):
            return False
        return (self.num_tokens == other.num_tokens and
                self.num_seqs == other.num_seqs and
                self.max_seq_len == other.max_seq_len and
                self.num_layers == other.num_layers)

    def __repr__(self):
        return (f"BatchDesc(tokens={self.num_tokens}, seqs={self.num_seqs}, "
                f"max_len={self.max_seq_len}, layers={self.num_layers})")


class CommandBufferCache:
    """
    Caches pre-recorded Vulkan command buffers keyed by batch shape.

    Usage:
        cache = CommandBufferCache(vulkan_ctx, max_cached=8)

        desc = BatchDescriptor(num_tokens=32, num_seqs=4, ...)

        if cache.has(desc):
            cache.replay(desc)  # Fast path: replay pre-recorded
        else:
            # Slow path: record new command buffer
            cache.begin_recording(desc)
            # ... record kernel launches ...
            cache.end_recording(desc)
    """

    def __init__(self, vulkan_ctx: Any, max_cached: int = 8):
        from vllm.vulkan import KernelLauncher

        self.ctx = vulkan_ctx
        self.max_cached = max_cached

        # Map batch descriptor -> pre-recorded launcher state
        self._cache: dict[BatchDescriptor, _CachedCommandBuffer] = {}
        self._lru: list[BatchDescriptor] = []

    def has(self, desc: BatchDescriptor) -> bool:
        """Check if a command buffer is cached for this batch shape."""
        return desc in self._cache

    def begin_recording(self, desc: BatchDescriptor):
        """Start recording a new command buffer for this batch shape."""
        from vllm.vulkan import KernelLauncher

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_cached and self._lru:
            evict_desc = self._lru.pop(0)
            if evict_desc in self._cache:
                del self._cache[evict_desc]
                logger.debug("Evicted cached command buffer for %s",
                             evict_desc)

        launcher = KernelLauncher(self.ctx.device)
        launcher.begin_recording()

        self._cache[desc] = _CachedCommandBuffer(launcher)
        self._lru.append(desc)
        logger.debug("Recording command buffer for %s", desc)

    def get_recording_launcher(self, desc: BatchDescriptor):
        """Get the launcher for recording commands."""
        if desc not in self._cache:
            raise RuntimeError(f"No recording in progress for {desc}")
        return self._cache[desc].launcher

    def end_recording(self, desc: BatchDescriptor):
        """Finish recording and mark as ready for replay."""
        if desc not in self._cache:
            raise RuntimeError(f"No recording in progress for {desc}")

        cached = self._cache[desc]
        # Don't submit yet — just mark as recorded
        cached.recorded = True
        logger.debug("Command buffer recorded for %s", desc)

    def replay(self, desc: BatchDescriptor):
        """Replay a pre-recorded command buffer (fast path)."""
        if desc not in self._cache:
            raise RuntimeError(f"No cached command buffer for {desc}")

        cached = self._cache[desc]
        if not cached.recorded:
            raise RuntimeError(f"Command buffer not fully recorded for {desc}")

        cached.launcher.submit_and_wait()

        # Move to end of LRU
        if desc in self._lru:
            self._lru.remove(desc)
        self._lru.append(desc)

    def invalidate(self, desc: BatchDescriptor | None = None):
        """
        Invalidate cached command buffers.

        Args:
            desc: Specific descriptor to invalidate, or None for all.
        """
        if desc is None:
            self._cache.clear()
            self._lru.clear()
            logger.debug("Invalidated all cached command buffers")
        elif desc in self._cache:
            del self._cache[desc]
            if desc in self._lru:
                self._lru.remove(desc)
            logger.debug("Invalidated cached command buffer for %s", desc)

    @property
    def size(self) -> int:
        return len(self._cache)


class _CachedCommandBuffer:
    """Internal: holds a pre-recorded launcher."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.recorded = False
