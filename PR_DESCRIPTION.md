# Fix SDAR/ENCODER_ONLY Models: Use Stable Fallback RoPE Kernel

## Summary

Fixes SDAR models producing gibberish/empty output after commit d8d0208. The new JIT RoPE kernel introduced in that commit has compatibility issues with ENCODER_ONLY attention patterns used in dLLM/block diffusion models like SDAR.

## Problem

After commit d8d0208, SDAR started producing:
- Gibberish tokens (random characters like `' "`)
- Empty token output
- Completely broken inference

Found by bisecting between working and broken commits - d8d0208 was the culprit.

## Investigation

The commit d8d0208 introduced a new JIT RoPE kernel to replace the FlashInfer-based implementation. After tracing through the code, here's what I found:

**What SDAR does differently:**
- Uses `AttentionType.ENCODER_ONLY` for block diffusion forward passes (dLLM-style)
- This is set in the RadixAttention initialization
- Most other models use default `AttentionType.DECODER`

**What I tried:**

1. **Disabled fused KV cache only** - Added check to skip `create_fused_set_kv_buffer_arg()` for ENCODER_ONLY. Didn't work - still got gibberish.

2. **Fixed tensor reshaping** - Changed `v.view_as(k)` to `v.reshape(k.shape)` in the fused kernel, tried ensuring contiguous tensors. Didn't work.

3. **Checked tensor shapes and strides** - Verified q, k, v shapes match what the kernel expects. Everything looked correct.

4. **Forced fallback kernel** - Made SDAR use the old `rotary_embedding` from `sglang.jit_kernel.pos_enc` instead of the new JIT kernel. This worked.

**Root cause:**
The new JIT RoPE kernel itself (not just the fused KV cache path) produces incorrect results for ENCODER_ONLY attention patterns. Even when calling `apply_rope_inplace()` without any fused KV cache arguments, the output is wrong.

The fallback kernel works fine because it uses a different, more tested implementation path.

## Solution

Added `set_fallback_kernel_for_encoder_only()` method to `RotaryEmbedding` class that forces use of the stable fallback kernel. SDAR calls this after creating its rotary embedding.

Also had to handle the case where fused KV cache is requested but fallback kernel is used - fallback doesn't support fused KV cache save, so we skip it and let the attention backend handle KV cache separately.

## Files Changed

**python/sglang/srt/layers/rotary_embedding/base.py**
- Added `set_fallback_kernel_for_encoder_only()` method
- Removed strict assertion in `forward_cuda()` that prevented fallback from working when fused KV cache arg is provided

**python/sglang/srt/models/sdar.py**
- Call `set_fallback_kernel_for_encoder_only()` in `SDARAttention.__init__()`
- Added `can_use_fused` check in `forward_prepare_native()` and `forward()` to skip fused KV cache when using fallback

**python/sglang/srt/models/sdar_moe.py**
- Same changes as sdar.py

## Benchmarks

MMLU: [FILL IN]%
GSM8K: [FILL IN]%

Should be similar to pre-d8d0208 numbers.

## Impact

- Only affects ENCODER_ONLY models (SDAR, SDAR-MoE)
- Other models (Llama, GPT-OSS, etc.) continue using the new JIT kernel with its optimizations
- Fallback kernel is stable and well-tested - used by XPU, NPU, CPU platforms

## Future Work

The real fix would be debugging why the new JIT kernel fails for ENCODER_ONLY patterns, but that requires deeper investigation into the CUDA kernel itself. This patch unblocks SDAR users in the meantime.
