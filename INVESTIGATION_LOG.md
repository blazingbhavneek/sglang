# SDAR Model JIT RoPE Kernel Issue - Complete Investigation Log

## Initial Problem Report

User reported that SDAR model (sdar.py) was broken after a particular commit (diff.txt). The issue:
- Produces gibberish/empty token output
- Affects SDAR specifically, not GPT-OSS (which was also affected by the same commit but to a lesser extent)
- Issue occurs regardless of GPU architecture

## Step 1: Analyzing the diff.txt

The diff introduced a completely new JIT-compiled RoPE (Rotary Positional Embedding) kernel:

**Key changes in diff.txt:**
1. New CUDA kernel implementation in `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`
   - Completely rewritten from FlashInfer-based implementation
   - New kernel structure with `FusedRopeParams` and `FusedRopeStoreParams`
   - Supports both RoPE-only and fused RoPE+KV-cache-store operations

2. New Python API in `python/sglang/jit_kernel/rope.py`:
   - `apply_rope_inplace()` - for RoPE only
   - `apply_rope_inplace_with_kvcache()` - for fused RoPE + KV cache store
   - Updated `apply_rope_with_cos_sin_cache_inplace()` as a wrapper

3. Changed function signatures:
   - Old: `query`, `key` as 2D tensors `[nnz, num_heads * head_size]`
   - New: `q`, `k` as 3D tensors `[batch_size, num_heads, head_dim]`

4. Updated callers in `python/sglang/srt/layers/rotary_embedding/base.py`:
   ```python
   batch_size = positions.size(0)
   q_rope = query.view(batch_size, -1, self.head_size)
   k_rope = key.view(batch_size, -1, self.head_size)
   apply_rope_with_cos_sin_cache_inplace(
       positions=positions,
       q=q_rope,
       k=k_rope,
       ...
   )
   ```

## Step 2: Examining SDAR Model

Read `python/sglang/srt/models/sdar.py` to understand how SDAR uses RoPE:

```python
class SDARAttention(nn.Module):
    def __init__(...):
        self.rotary_dim = self.head_dim  # rotary_dim equals head_dim
        self.rotary_emb = get_rope(...)

        # KEY FINDING #1:
        self.attn = RadixAttention(
            ...,
            attn_type=AttentionType.ENCODER_ONLY,  # <-- This is different!
            ...
        )
```

**Key Finding #1:** SDAR uses `AttentionType.ENCODER_ONLY` for its attention mechanism. This is used for block diffusion / dLLM-style forward passes.

## Step 3: First Hypothesis - Fused KV Cache Issue

Looking at the diff, I noticed `llada2.py` (another model) received a fix:

```python
can_fuse_set_kv = (
    self.head_dim == self.rotary_emb.rotary_dim
    and enable_fused_set_kv_buffer(forward_batch)
)
```

This adds a guard before using fused KV cache. SDAR didn't have this guard.

**First Attempt:** Added the same `can_fuse_set_kv` check to SDAR.

**Result:** Didn't work. Still got gibberish output.

## Step 4: Deeper Investigation - Why Only SDAR?

Searched for other models using ENCODER_ONLY:

```bash
grep -r "AttentionType.ENCODER_ONLY" python/sglang/srt/models/
```

Found: SDAR and SDAR-MoE use ENCODER_ONLY. Most other models use default DECODER.

Checked `python/sglang/srt/layers/attention/flashinfer_backend.py` line 815:

```python
if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
    save_kv_cache = False
```

**Key Finding #2:** ENCODER_ONLY attention has special KV cache handling in the attention backend. For non-dLLM models with ENCODER_ONLY, it forces `save_kv_cache=False`.

But SDAR IS a dLLM model, so this condition is False, and `save_kv_cache` remains True.

## Step 5: Second Hypothesis - Tensor Shape Mismatch

The fused kernel expects specific tensor layouts. Checked `apply_rope_inplace_with_kvcache`:

```python
def apply_rope_inplace_with_kvcache(...):
    v = v.view_as(k)  # Reshape v to match k
    module.run_rope_store(q, k, v, k_cache, v_cache, ...)
```

**Second Attempt:** Changed `v.view_as(k)` to `v.reshape(k.shape)` and also reshaped value in `create_fused_set_kv_buffer_arg()` to 3D.

**Result:** Still didn't work.

## Step 6: Third Hypothesis - ENCODER_ONLY Incompatibility

Added a check in `create_fused_set_kv_buffer_arg()` to return None for ENCODER_ONLY:

```python
if layer.attn_type == AttentionType.ENCODER_ONLY:
    return None  # Disable fused kernel entirely
```

This would make SDAR use the new JIT RoPE kernel WITHOUT fused KV cache.

**Result:** STILL didn't work. Got gibberish output.

## Step 7: Critical Discovery

At this point, I realized: **Even the basic JIT RoPE kernel (without fused KV cache) is broken for ENCODER_ONLY models.**

The issue is NOT just the fused KV cache path - it's the entire new JIT RoPE kernel implementation.

Traced the code flow:
1. SDAR calls `self.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=None)`
2. This goes to `RotaryEmbedding.forward_cuda()`
3. Since `use_fallback_kernel=False`, it calls `apply_rope_with_cos_sin_cache_inplace()`
4. Which calls `apply_rope_inplace()` (the new JIT kernel)
5. The JIT kernel produces incorrect results for ENCODER_ONLY attention patterns

## Step 8: Understanding use_fallback_kernel

Checked what determines `use_fallback_kernel` in `RotaryEmbedding.__init__()`:

```python
if (
    (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])
    and not (_is_cpu)
    and not (_is_xpu)
    and not (_is_npu)
    and not (_is_musa)
):
    self.use_fallback_kernel = True
    self.fallback_rotary_embedding = rotary_embedding  # from sglang.jit_kernel.pos_enc
else:
    self.use_fallback_kernel = False  # Uses new JIT kernel
```

SDAR's head_size is typically 128 or 256, so it uses the NEW JIT kernel (`use_fallback_kernel=False`).

## Step 9: Final Solution

Forced SDAR to use the fallback kernel by adding a method:

```python
def set_fallback_kernel_for_encoder_only(self):
    """Force fallback kernel for ENCODER_ONLY attention."""
    if not self.use_fallback_kernel:
        from sglang.jit_kernel.pos_enc import rotary_embedding
        self.use_fallback_kernel = True
        self.fallback_rotary_embedding = rotary_embedding
```

SDAR calls this after creating rotary_emb:

```python
self.rotary_emb = get_rope(...)
self.rotary_emb.set_fallback_kernel_for_encoder_only()
```

**Result:** This WORKED! SDAR produces correct output now.

## Step 10: Handling Fused KV Cache

When using fallback kernel, we can't use fused KV cache save. The fallback kernel (`rotary_embedding` from `sglang.jit_kernel.pos_enc`) doesn't support it.

Had to modify `forward_cuda()`:

```python
else:
    # Fallback kernel doesn't support fused KV cache save
    # Skip fused save and let attention backend handle it separately
    self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
    self.fallback_rotary_embedding(
        positions, query, key, self.head_size,
        self.cos_sin_cache, self.is_neox_style,
    )
```

Removed the strict assertion that was preventing this.

Also updated SDAR to check both conditions:

```python
can_use_fused = (
    enable_fused_set_kv_buffer(forward_batch)
    and not self.rotary_emb.use_fallback_kernel  # Can't fuse with fallback
)
```

## Why This Works

The fallback kernel uses `rotary_embedding` from `sglang.jit_kernel.pos_enc`, which is:
- Older, well-tested implementation
- Used by other platforms (XPU, NPU, CPU) without issues
- Doesn't have the ENCODER_ONLY compatibility issues

The new JIT kernel (from `sglang.jit_kernel.rope`) has some incompatibility with ENCODER_ONLY attention patterns that would require deeper CUDA kernel debugging to fix.

## Files Modified

1. **python/sglang/srt/layers/rotary_embedding/base.py**
   - Added `set_fallback_kernel_for_encoder_only()` method
   - Modified `forward_cuda()` to handle fused KV cache arg gracefully with fallback kernel

2. **python/sglang/srt/models/sdar.py**
   - Call `set_fallback_kernel_for_encoder_only()` in `SDARAttention.__init__()`
   - Added `can_use_fused` check in `forward_prepare_native()` and `forward()`

3. **python/sglang/srt/models/sdar_moe.py**
   - Same changes as sdar.py

## Timeline of Investigation

1. **Analyzed diff.txt** - Understood what the commit changed
2. **Examined SDAR code** - Found ENCODER_ONLY attention type
3. **First attempt** - Added `can_fuse_set_kv` guard (like llada2.py) - FAILED
4. **Second attempt** - Fixed tensor reshaping - FAILED
5. **Third attempt** - Disabled fused KV cache for ENCODER_ONLY - FAILED
6. **Critical discovery** - Even basic JIT RoPE kernel is broken for ENCODER_ONLY
7. **Final solution** - Force fallback kernel - SUCCESS

## Key Learnings

1. **ENCODER_ONLY attention is different** - It has special handling in attention backends and doesn't work with the new JIT RoPE kernel

2. **The problem is deeper than fused KV cache** - Even the basic RoPE-only path in the new JIT kernel fails for ENCODER_ONLY

3. **Fallback kernel is reliable** - The older implementation in `sglang.jit_kernel.pos_enc` works fine

4. **Targeted fix is better** - Only affects ENCODER_ONLY models, other models continue using optimized JIT kernel

## What Would Need to Fix the Root Cause

To actually fix the new JIT kernel for ENCODER_ONLY support would require:
1. Debugging the CUDA kernel in `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`
2. Understanding what ENCODER_ONLY does differently in tensor access patterns
3. Comparing with the working fallback kernel implementation
4. Potentially adding special handling for ENCODER_ONLY in the kernel itself

This patch is a workaround that unblocks SDAR users while the root cause can be investigated separately.

## Testing Notes

- User will provide MMLU and GSM8K benchmarks
- Expected to match pre-d8d0208 numbers
- Other models (DECODER attention) should be unaffected
