"""
debug_hook.py — Verify that forward hooks modify Llama layer outputs
=====================================================================
Runs the model on ONE prompt twice: once clean, once with a hook that
subtracts a random direction. Compares logits. If identical, the hook
is broken. If different, the hook works and the intervention script
just needs the right hook implementation.
 
Usage:
  python scripts/debug_hook.py
"""
 
import torch
import numpy as np
from pathlib import Path
 
MODEL_PATH = Path("/Users/brandonlepine/Repositories/Research_Repositories/smi/models/llama2-13b")
 
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
 
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16
    ).to("mps")
    model.eval()
 
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
 
    target_layer = 20
    direction = torch.randn(5120, dtype=torch.float16, device="mps")
    direction = direction / direction.norm()
 
    # --- Run 1: Clean (no hook) ---
    with torch.no_grad():
        out_clean = model(**inputs)
        logits_clean = out_clean.logits[0, -1, :].clone()
 
    # --- Attempt A: forward_hook modifying output in-place ---
    call_count_a = [0]
    def hook_a(module, args, output):
        call_count_a[0] += 1
        if isinstance(output, tuple):
            output[0].sub_(4.0 * direction.unsqueeze(0))
        elif hasattr(output, '__getitem__'):
            output[0].sub_(4.0 * direction.unsqueeze(0))
        return output
 
    h = model.model.layers[target_layer].register_forward_hook(hook_a)
    with torch.no_grad():
        out_a = model(**inputs)
        logits_a = out_a.logits[0, -1, :].clone()
    h.remove()
 
    diff_a = (logits_clean - logits_a).abs().max().item()
    print(f"\nAttempt A (in-place .sub_ on output[0]):")
    print(f"  Hook called: {call_count_a[0]} times")
    print(f"  Max logit diff: {diff_a:.6f}")
    print(f"  {'WORKING' if diff_a > 0.01 else 'BROKEN'}")
 
    # --- Attempt B: forward_hook returning new tuple ---
    call_count_b = [0]
    def hook_b(module, args, output):
        call_count_b[0] += 1
        if isinstance(output, tuple):
            hidden = output[0] - 4.0 * direction.unsqueeze(0)
            return (hidden,) + output[1:]
        return output
 
    h = model.model.layers[target_layer].register_forward_hook(hook_b)
    with torch.no_grad():
        out_b = model(**inputs)
        logits_b = out_b.logits[0, -1, :].clone()
    h.remove()
 
    diff_b = (logits_clean - logits_b).abs().max().item()
    print(f"\nAttempt B (return new tuple):")
    print(f"  Hook called: {call_count_b[0]} times")
    print(f"  Max logit diff: {diff_b:.6f}")
    print(f"  {'WORKING' if diff_b > 0.01 else 'BROKEN'}")
 
    # --- Attempt C: pre-hook on NEXT layer ---
    call_count_c = [0]
    def pre_hook_c(module, args):
        call_count_c[0] += 1
        # args[0] is the hidden_states input to this layer
        if isinstance(args, tuple) and len(args) > 0:
            hidden = args[0]
            hidden.sub_(4.0 * direction.unsqueeze(0))
        return args
 
    next_layer = target_layer + 1
    h = model.model.layers[next_layer].register_forward_pre_hook(pre_hook_c)
    with torch.no_grad():
        out_c = model(**inputs)
        logits_c = out_c.logits[0, -1, :].clone()
    h.remove()
 
    diff_c = (logits_clean - logits_c).abs().max().item()
    print(f"\nAttempt C (pre-hook on layer {next_layer}):")
    print(f"  Hook called: {call_count_c[0]} times")
    print(f"  Max logit diff: {diff_c:.6f}")
    print(f"  {'WORKING' if diff_c > 0.01 else 'BROKEN'}")
 
    # --- Attempt D: hook on self_attn of next layer (modify residual input) ---
    call_count_d = [0]
    def pre_hook_d(module, args):
        call_count_d[0] += 1
        if isinstance(args, tuple) and len(args) > 0:
            if isinstance(args[0], torch.Tensor):
                args[0].sub_(4.0 * direction.unsqueeze(0))
        return args
 
    h = model.model.layers[next_layer].self_attn.register_forward_pre_hook(pre_hook_d)
    with torch.no_grad():
        out_d = model(**inputs)
        logits_d = out_d.logits[0, -1, :].clone()
    h.remove()
 
    diff_d = (logits_clean - logits_d).abs().max().item()
    print(f"\nAttempt D (pre-hook on layer {next_layer}.self_attn):")
    print(f"  Hook called: {call_count_d[0]} times")
    print(f"  Max logit diff: {diff_d:.6f}")
    print(f"  {'WORKING' if diff_d > 0.01 else 'BROKEN'}")
 
    # --- Summary ---
    print("\n" + "=" * 50)
    working = []
    for name, diff in [("A", diff_a), ("B", diff_b), ("C", diff_c), ("D", diff_d)]:
        if diff > 0.01:
            working.append(name)
    if working:
        print(f"  USE ATTEMPT {working[0]} — it modifies activations successfully")
    else:
        print("  ALL HOOKS BROKEN — need a different strategy")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()