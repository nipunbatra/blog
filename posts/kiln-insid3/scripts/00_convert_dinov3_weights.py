"""Step 0 (env) — Convert HF-format DINOv3 ViT-L/16 weights to the original
torch.hub key layout that INSID3 expects.

The only non-gated DINOv3 checkpoints are in HuggingFace-transformers key format
(`embeddings.*`, `layer.N.attention.{q,k,v}_proj`). INSID3 loads the *original*
Meta hub model (`cls_token`, `blocks.N.attn.qkv`, strict=True). We:

  1. Build the hub arch with pretrained=False -> correct keys + deterministic
     buffers (rope_embed.periods, qkv.bias_mask) from init_weights().
  2. Remap the learned HF keys onto original names (q/k/v concatenated -> qkv;
     k has no bias in HF, and the original masks the k-bias anyway -> zeros).
  3. Merge over the init state_dict (keeps buffers), load strict=True, re-save.

Run inside the INSID3 venv with TORCH_HOME set. Functionally verified by the
cat demo (a wrong q/k/v order destroys attention -> wrong segmentation).
"""
import torch

SRC = "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"   # HF-format (in place)
DST = SRC                                                       # overwrite with original-format
DEPTH = 24

print("[1] build arch (pretrained=False) for correct keys + buffers")
m = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16",
                   pretrained=False, source="github")
sd = m.state_dict()                       # original keys, init values

print("[2] load HF-format learned weights")
hf = torch.load(SRC, map_location="cpu")
if "state_dict" in hf:
    hf = hf["state_dict"]

out = {k: v.clone() for k, v in sd.items()}   # start from init (keeps buffers)

# --- non-block tensors ---
direct = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "storage_tokens",
    "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
}
for hk, ok in direct.items():
    src = hf[hk]
    assert out[ok].numel() == src.numel(), (ok, out[ok].shape, src.shape)
    out[ok] = src.reshape(out[ok].shape).clone()   # tokens may differ by a singleton dim

# --- per-block ---
for i in range(DEPTH):
    L, B = f"layer.{i}", f"blocks.{i}"
    # norms, layerscales, mlp, output proj  (direct)
    pairs = {
        f"{B}.norm1.weight": f"{L}.norm1.weight", f"{B}.norm1.bias": f"{L}.norm1.bias",
        f"{B}.norm2.weight": f"{L}.norm2.weight", f"{B}.norm2.bias": f"{L}.norm2.bias",
        f"{B}.ls1.gamma": f"{L}.layer_scale1.lambda1",
        f"{B}.ls2.gamma": f"{L}.layer_scale2.lambda1",
        f"{B}.mlp.fc1.weight": f"{L}.mlp.up_proj.weight", f"{B}.mlp.fc1.bias": f"{L}.mlp.up_proj.bias",
        f"{B}.mlp.fc2.weight": f"{L}.mlp.down_proj.weight", f"{B}.mlp.fc2.bias": f"{L}.mlp.down_proj.bias",
        f"{B}.attn.proj.weight": f"{L}.attention.o_proj.weight",
        f"{B}.attn.proj.bias": f"{L}.attention.o_proj.bias",
    }
    for ok, hk in pairs.items():
        assert out[ok].shape == hf[hk].shape, (ok, out[ok].shape, hf[hk].shape)
        out[ok] = hf[hk].clone()
    # fused qkv: concat along output dim
    qw, kw, vw = (hf[f"{L}.attention.{p}_proj.weight"] for p in "qkv")
    out[f"{B}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0).clone()
    qb = hf[f"{L}.attention.q_proj.bias"]
    vb = hf[f"{L}.attention.v_proj.bias"]
    kb = torch.zeros_like(qb)              # HF k_proj has no bias; original masks it
    out[f"{B}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0).clone()

print("[3] strict load + save")
missing, unexpected = m.load_state_dict(out, strict=False)
# only the deterministic buffers may differ from HF; nothing should be missing/unexpected here
assert not missing and not unexpected, (missing[:5], unexpected[:5])
m.load_state_dict(out, strict=True)        # must pass
torch.save(out, DST)
print(f"[done] wrote original-format checkpoint -> {DST}  ({len(out)} keys)")
