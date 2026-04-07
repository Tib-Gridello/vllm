# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer outlier channel calibration for TurboQuant.

Usage:
    python -m vllm.v1.attention.ops.turboquant_calibrate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output calibration.pt \
        --outlier-ratio 0.25 \
        --num-samples 32

Output: a dict with per-layer outlier masks saved as a .pt file:
    {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "outlier_ratio": 0.25,
        "head_dim": 128,
        "num_layers": 28,
        "masks": {
            0: tensor([True, False, ...]),  # [head_dim] per layer
            1: tensor([...]),
            ...
        },
        "channel_variances": {
            0: tensor([...]),  # [head_dim] per-channel variance
            ...
        },
    }
"""

import argparse

import torch

from vllm.v1.attention.ops.turboquant import calibrate_outlier_channels


def calibrate_model(
    model_name: str,
    output_path: str,
    outlier_ratio: float = 0.25,
    num_samples: int = 32,
    max_tokens: int = 512,
    device: str = "cuda",
) -> dict:
    """Run calibration on a model and save per-layer outlier masks.

    Uses vLLM to run inference and captures K/V projections
    at each attention layer via hooks.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Build calibration prompts
    prompts = [
        "The theory of general relativity describes gravity as the "
        "curvature of spacetime caused by mass and energy.",
        "In quantum mechanics, particles can exist in superposition "
        "states until measured, collapsing the wave function.",
        "Machine learning models learn patterns from data through "
        "optimization of loss functions via gradient descent.",
        "The human genome contains approximately three billion base "
        "pairs organized into 23 pairs of chromosomes.",
    ] * (num_samples // 4 + 1)
    prompts = prompts[:num_samples]

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    # Hook to capture K/V projections
    k_activations = {}  # layer_idx -> list of tensors
    v_activations = {}
    hooks = []

    # Find attention layers and hook into K/V projections
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )

    print(
        f"Model: {num_layers} layers, head_dim={head_dim}, num_kv_heads={num_kv_heads}"
    )

    # Hook into k_proj and v_proj linear layers
    for layer_idx in range(num_layers):
        k_activations[layer_idx] = []
        v_activations[layer_idx] = []

        # Find the attention module — handle different model architectures
        attn = None
        if hasattr(model, "model"):
            layers = model.model.layers
        elif hasattr(model, "transformer"):
            layers = model.transformer.h
        else:
            raise RuntimeError(f"Unknown model architecture: {type(model)}")

        layer = layers[layer_idx]
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
        elif hasattr(layer, "attention"):
            attn = layer.attention
        else:
            raise RuntimeError(f"Can't find attention in layer {layer_idx}")

        def make_k_hook(idx):
            def hook_fn(module, input_args, output):
                # output shape: [batch, seq_len, num_kv_heads * head_dim]
                with torch.no_grad():
                    out = output.detach().float()
                    # Reshape to [batch * seq_len, num_kv_heads, head_dim]
                    out = out.reshape(-1, num_kv_heads, head_dim)
                    # Sample at most 1000 tokens per batch
                    if out.shape[0] > 1000:
                        perm = torch.randperm(out.shape[0])[:1000]
                        out = out[perm]
                    k_activations[idx].append(out.cpu())

            return hook_fn

        def make_v_hook(idx):
            def hook_fn(module, input_args, output):
                with torch.no_grad():
                    out = output.detach().float()
                    out = out.reshape(-1, num_kv_heads, head_dim)
                    if out.shape[0] > 1000:
                        perm = torch.randperm(out.shape[0])[:1000]
                        out = out[perm]
                    v_activations[idx].append(out.cpu())

            return hook_fn

        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if k_proj is None or v_proj is None:
            raise RuntimeError(f"Can't find k_proj/v_proj in layer {layer_idx}")
        hooks.append(k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hooks.append(v_proj.register_forward_hook(make_v_hook(layer_idx)))

    # Run calibration inference
    print(f"Running calibration with {num_samples} samples...")
    with torch.no_grad():
        # Process in small batches to avoid OOM
        batch_size = 4
        for i in range(0, len(prompts), batch_size):
            batch_inputs = {k: v[i : i + batch_size] for k, v in inputs.items()}
            model(**batch_inputs)
            if (i // batch_size) % 4 == 0:
                print(f"  Processed {i + batch_size}/{len(prompts)} samples")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute per-layer outlier masks
    print("Computing outlier masks...")
    result = {
        "model": model_name,
        "outlier_ratio": outlier_ratio,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "masks": {},
        "channel_variances": {},
    }

    for layer_idx in range(num_layers):
        # Combine K and V activations for this layer
        k_all = torch.cat(k_activations[layer_idx], dim=0)
        v_all = torch.cat(v_activations[layer_idx], dim=0)
        combined = torch.cat([k_all, v_all], dim=0)

        # Compute per-channel variance
        var = combined.float().var(dim=(0, 1))  # [head_dim]
        mask = calibrate_outlier_channels(combined, outlier_ratio)

        result["masks"][layer_idx] = mask
        result["channel_variances"][layer_idx] = var

        n_out = mask.sum().item()
        top_var = var[mask].mean().item()
        bot_var = var[~mask].mean().item()
        if layer_idx % 7 == 0:
            print(
                f"  Layer {layer_idx:2d}: {n_out} outlier channels, "
                f"outlier_var={top_var:.4f}, regular_var={bot_var:.4f}, "
                f"ratio={top_var / (bot_var + 1e-10):.1f}x"
            )

    # Save
    torch.save(result, output_path)
    print(f"Saved calibration to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant per-layer calibration")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="tq_calibration.pt")
    parser.add_argument("--outlier-ratio", type=float, default=0.25)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    calibrate_model(
        args.model,
        args.output,
        outlier_ratio=args.outlier_ratio,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
    )
