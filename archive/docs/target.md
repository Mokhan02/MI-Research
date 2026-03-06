# Phase 1: SAE Target Selection

## ✅ Phase 1 Locked Target

**Status:** PROVISIONALLY LOCKED - Requires verification before experiments  
**Config File:** `configs/targets/gemma2_2b_gemmascope_res16k.yaml`

**What "Locked" Means:** Target is chosen, but NOT fully verified. You must complete all 3 verifications below before running experiments.

**Model:** `google/gemma-2-2b`  
**SAE Set:** `gemmascope-res-16k` (GemmaScope resid_post SAEs, 16k features)  
**SAE Source:** Neuronpedia  
**SAE ID:** `TBD` (TODO: Get specific SAE ID from Neuronpedia for layer 20)  
**Hook Point:** `blocks.20.hook_resid_post` (provisional - to be confirmed)  
**Layer Index:** `20` (provisional - to be confirmed)  
**d_model:** `2048` (CONFIRM: verify matches model.config.hidden_size)  
**n_features:** `16384` (CONFIRM: verify matches SAE dictionary size)  
**Weights Repo:** `google/gemma-scope` (CONFIRM: verify this is the correct HuggingFace repo)

### Why This Target

- **Feasible on A10 GPU:** ~2B params, ~4-5GB VRAM for inference, leaves plenty of headroom
- **Fast iteration:** Smaller model = faster forward passes, quicker experiments
- **Publicly accessible:** Gemma-2-2b-it is publicly available on Hugging Face
- **Good SAE availability:** Neuronpedia likely has multiple SAE checkpoints for Gemma-2-2b
- **Standard architecture:** Gemma uses standard transformer blocks, hook points are straightforward
- **Sufficient complexity:** 2B params is large enough to have meaningful features while being manageable

### Hook Point Details

**Structure:** Gemma-2 uses standard transformer architecture:
```
model.layers.{i}
  ├── self_attn (attention output)
  ├── mlp (MLP output)
  └── (resid_post = attention + MLP residual)
```

**Hook Point:** `blocks.20.hook_resid_post` (provisional - **DO NOT GUESS, must come from SAE metadata**)
- This is **TransformerLens-style naming** - HuggingFace models do NOT expose this name
- If SAE was trained with TransformerLens, you'll need a hook resolver to map to HF module outputs
- If SAE was trained on HF module outputs, use the actual HF module name
- **Must check SAE metadata** to determine which naming convention was used
- Expected output shape: `(batch, seq_len, 2048)`

**⚠️ CRITICAL:** Hook point naming must come from SAE release metadata (Neuronpedia/GemmaScope), not guessed from model structure.

**Neuronpedia Link:** `[TO BE FILLED - e.g., https://www.neuronpedia.org/gemma-2-2b/{sae_id}]`

### Phase 1 Verification Checklist

**Run verification script:** `uv run python scripts/verify_target.py`

#### ✅ Verification 1: Model Loading and Module Listing

**Automated check** - Run this first:

```bash
uv run python scripts/verify_target.py
```

Or run manually:
```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto")
names = [n for n,_ in m.named_modules()]
print("num modules:", len(names))
for pat in ["layers.20", "model.layers.20", "blocks.20", "attn", "mlp"]:
    hits = [n for n in names if pat in n][:15]
    print(f"\nPATTERN: {pat} hits: {len(hits)}")
    for h in hits:
        print(f"  {h}")
```

**What to check:**
- [ ] Model loads successfully
- [ ] `model.config.hidden_size == 2048` (d_model)
- [ ] Layer 20 exists (check module names for layer 20)
- [ ] **Note:** `blocks.20.hook_resid_post` will NOT exist in HF modules (it's TransformerLens naming)
- [ ] If hook point is TransformerLens-style, you'll need a hook resolver to map to actual HF module outputs

**Status:** ⚠️ If `blocks.20.hook_resid_post` doesn't exist, that's expected - you need SAE metadata to determine actual hook point.

---

#### ✅ Verification 2: Hook Point Naming from SAE Metadata

**⚠️ MANUAL VERIFICATION - DO NOT GUESS**

**What to check:**
- [ ] **Check SAE metadata** (Neuronpedia page, GemmaScope tooling, SAE checkpoint README)
- [ ] **Determine:** Was SAE trained with TransformerLens naming or HuggingFace module outputs?
- [ ] **If TransformerLens:** Hook point is logical (`blocks.20.hook_resid_post`) - implement hook resolver
- [ ] **If HuggingFace:** Hook point is actual module name (e.g., `model.layers.20`) - use directly
- [ ] **Update config:** Fill in confirmed hook point name in `configs/targets/gemma2_2b_gemmascope_res16k.yaml`

**Sources to check:**
- Neuronpedia SAE page metadata
- GemmaScope repository documentation
- SAE checkpoint config files or README
- SAE training code/comments

**Current provisional:** `blocks.20.hook_resid_post` (TransformerLens-style, needs confirmation)

---

#### ✅ Verification 3: Decoder Shape Confirmation

**⚠️ MANUAL VERIFICATION - REQUIRED BEFORE EXPERIMENTS**

**What to check:**
- [ ] Load SAE decoder weights from `google/gemma-scope` repo
- [ ] **Confirm decoder shape is `(16384, 2048)`** i.e., `(n_features, d_model)`
  - OR `(2048, 16384)` if transposed - handle accordingly
- [ ] **Confirm `decoder.shape[-1] == 2048`** (d_model dimension matches model)
- [ ] **Confirm `decoder.shape[0] == 16384`** (n_features matches SAE dictionary size)
- [ ] Add shape assertions to `load_sae()` in `src/model_utils.py`

**Verification code to add:**
```python
# In src/model_utils.py load_sae()
decoder = load_decoder_weights(...)  # Load from SAE checkpoint
assert decoder.shape == (16384, 2048) or decoder.shape == (2048, 16384), \
    f"Expected decoder shape (16384, 2048) or (2048, 16384), got {decoder.shape}"
if decoder.shape[0] != 16384:
    decoder = decoder.T  # Transpose if needed
assert decoder.shape == (16384, 2048), f"After transpose, expected (16384, 2048), got {decoder.shape}"
assert decoder.shape[-1] == 2048, f"d_model dimension must be 2048, got {decoder.shape[-1]}"
```

**⚠️ No confirmation = no experiment.** Do not proceed until decoder shape is verified.

---

## Alternative Candidates

### Candidate 2: Gemma-2-7b-it

**Model:** `google/gemma-2-7b-it`  
**Expected d_model:** `3072` or `4096`  
**Hook Point:** `model.layers.{layer_idx}` (resid_post)  
**Expected n_features:** `[TO BE FILLED - typically 16384-65536]`

**Pros:**
- More interesting features (larger model = richer representations)
- Still publicly accessible
- Better for demonstrating steerability effects (more capacity)

**Cons:**
- Tighter on A10 GPU (~14GB VRAM, less headroom)
- Slower iteration (larger model = slower forward passes)
- May need gradient checkpointing or optimizations

**When to use:** If Gemma-2-2b proves too simple or if GPU headroom allows.

---

### Candidate 3: Gemma-2-9b-it

**Model:** `google/gemma-2-9b-it`  
**Expected d_model:** `4096`  
**Hook Point:** `model.layers.{layer_idx}` (resid_post)  
**Expected n_features:** `[TO BE FILLED - typically 32768+]`

**Pros:**
- Most feature-rich option
- Best for demonstrating complex steerability patterns

**Cons:**
- Likely too large for A10 GPU (~18GB VRAM, may not fit)
- Slowest iteration time
- May require quantization or model sharding

**When to use:** Only if A10 has sufficient VRAM or if using larger GPU.

---

## Implementation Checklist (Legacy - See Phase 1 Checklist Above)

*Note: Phase 1 target is now locked. See "Phase 1 Verification Checklist" above for current status.*

Legacy checklist items (for reference):
- [ ] **SAE ID:** `[Neuronpedia SAE identifier]` → Now: TBD, to be filled
- [ ] **Neuronpedia URL:** `[Direct link to SAE page]` → To be filled
- [ ] **Layer index:** `20` → Provisional, to be confirmed
- [ ] **Hook point:** `blocks.20.hook_resid_post` → Provisional, to be confirmed
- [ ] **d_model:** `2048` → CONFIRM
- [ ] **n_features:** `16384` → CONFIRM
- [ ] **SAE checkpoint format:** `[How to load: .pt, .safetensors, API, etc.]` → To be determined
- [ ] **Decoder weights shape:** `(16384, 2048)` → CONFIRM
- [ ] **Normalization:** `[Are decoder weights already normalized?]` → To be determined

## Next Steps (Phase 1)

**Order of operations:**

1. **Run Verification 1:** `uv run python scripts/verify_target.py` (or manual model loading)
   - Confirm model loads
   - List modules to see actual naming convention
   - Note that TransformerLens-style names won't exist in HF models

2. **Run Verification 2:** Check SAE metadata for hook point naming
   - Do NOT guess hook point name
   - Check Neuronpedia/GemmaScope metadata
   - Determine if SAE uses TransformerLens or HuggingFace naming
   - Update config with confirmed hook point

3. **Run Verification 3:** Load SAE decoder weights and confirm shape
   - Load decoder from `google/gemma-scope` repo
   - Verify shape is `(16384, 2048)` or transposed
   - Add shape assertions to `load_sae()` code
   - **Do not proceed until this is confirmed**

4. **Implement hook resolver** (if needed):
   - If hook point is TransformerLens-style, create resolver to map to HF module outputs
   - Test with `scripts/99_hook_sanity.py`

5. **Update config:** Fill in all confirmed values in `configs/targets/gemma2_2b_gemmascope_res16k.yaml`

6. **Run hook sanity:** Use `scripts/99_hook_sanity.py` to verify steering works end-to-end

**Only after all 3 verifications pass can you run the full pipeline.**

## Notes

- Gemma-2 models use `torch.float32` by default, but can use `bfloat16` for memory savings
- For A10 GPU, consider `dtype: "bfloat16"` in config if memory is tight
- Hook point verification should be done before running full pipeline
- SAE decoder weights typically need to be normalized (check SAE metadata)
