"""Un Qwen3.5 diminuto (misma arquitectura, pesos aleatorios) y un processor simulado, para probar el
camino de entrenamiento e inferencia en CPU sin descargar nada.

El processor reproduce lo que importa del real (Qwen3VLProcessor): plantilla ChatML con
`enable_thinking`, expansión de <|image_pad|> a grid_thw.prod() / merge² tokens, `mm_token_type_ids`
por token, relleno a la izquierda y `batch_decode`. El tokenizador es por caracteres.
"""

from __future__ import annotations

import re

import torch

SPECIALS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
            "<|endoftext|>"]
_SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")


class CharTokenizer:
    def __init__(self):
        self.vocab = {s: i for i, s in enumerate(SPECIALS)}
        for c in [chr(i) for i in range(32, 127)] + ["\n"]:
            self.vocab[c] = len(self.vocab)
        self.inv = {i: s for s, i in self.vocab.items()}
        self.pad_token, self.eos_token = "<|pad|>", "<|endoftext|>"
        self.pad_token_id, self.eos_token_id = self.vocab["<|pad|>"], self.vocab["<|endoftext|>"]
        self.padding_side = "left"
        self.chat_template = None

    def encode(self, s: str) -> list[int]:
        ids = []
        for part in _SPLIT.split(s):
            if part in self.vocab and part in SPECIALS:
                ids.append(self.vocab[part])
            else:
                ids.extend(self.vocab.get(c, self.vocab["?"]) for c in part)
        return ids

    def __call__(self, text, add_special_tokens=False, **_):
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": [self.encode(t) for t in text]}

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]

    def decode(self, ids, skip_special_tokens=True, **_):
        out = [self.inv[int(i)] for i in ids]
        if skip_special_tokens:
            out = [t for t in out if t not in SPECIALS]
        return "".join(out)


class MockQwenProcessor:
    def __init__(self, image_processor, merge: int = 2):
        self.tokenizer = CharTokenizer()
        self.image_processor = image_processor
        self.merge = merge
        self.chat_template = "chatml-mock"

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, enable_thinking=None, **_):
        s = ""
        for m in msgs:
            s += f"<|im_start|>{m['role']}\n"
            for c in m["content"]:
                s += "<|vision_start|><|image_pad|><|vision_end|>" if c["type"] == "image" else c["text"]
            s += "<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")
        return s

    def __call__(self, text, images=None, padding=False, return_tensors="pt", **_):
        from transformers import BatchFeature

        ip = self.image_processor(images=images, return_tensors="pt")
        grid = ip["image_grid_thw"]
        k, seqs = 0, []
        for t in text:
            parts = t.split("<|image_pad|>")
            s = parts[0]
            for p in parts[1:]:
                s += "<|image_pad|>" * int(grid[k].prod() // self.merge ** 2) + p
                k += 1
            seqs.append(self.tokenizer.encode(s))
        L = max(map(len, seqs))
        if len(seqs) > 1 and not padding and len(set(map(len, seqs))) > 1:
            raise ValueError("se necesita padding")
        pad = self.tokenizer.pad_token_id
        img_id = self.tokenizer.vocab["<|image_pad|>"]
        ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs])
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
        mm = (ids == img_id).long()
        return BatchFeature({"input_ids": ids, "attention_mask": att, "pixel_values": ip["pixel_values"],
                             "image_grid_thw": grid, "mm_token_type_ids": mm})

    def batch_decode(self, ids, skip_special_tokens=True, **_):
        return [self.tokenizer.decode(r, skip_special_tokens) for r in ids]


def tiny_qwen35(dtype=torch.bfloat16):
    import transformers as T

    tok = CharTokenizer()
    c = T.Qwen3_5Config()
    v = c.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 1, 32, 64, 2
    v.out_hidden_size, v.num_position_embeddings = 64, 64
    tc = c.text_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 4
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
    tc.linear_key_head_dim, tc.linear_value_head_dim = 16, 16
    tc.linear_num_key_heads, tc.linear_num_value_heads = 2, 4
    tc.vocab_size = len(tok.vocab)
    tc.pad_token_id = tok.pad_token_id
    c.image_token_id = tok.vocab["<|image_pad|>"]
    c.vision_start_token_id = tok.vocab["<|vision_start|>"]
    c.vision_end_token_id = tok.vocab["<|vision_end|>"]
    c.video_token_id = tok.vocab["<|endoftext|>"]            # no se usa
    torch.manual_seed(0)
    model = T.Qwen3_5ForConditionalGeneration(c).to(dtype).eval()
    model.generation_config.eos_token_id = [tok.vocab["<|im_end|>"], tok.eos_token_id]
    model.generation_config.pad_token_id = tok.pad_token_id
    return model


def tiny_processor():
    import transformers as T

    ip = T.Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                                 size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64})
    return MockQwenProcessor(ip)


def tiny_describer(max_new_tokens: int = 24):
    """QwenVLDescriber listo para usar, sin `load()` (no hay pesos que descargar)."""
    from vlmfid.config import _load_model
    from vlmfid.models.qwen_vl import QwenVLDescriber

    mcfg = _load_model("qwen35")
    mcfg.quant = "bf16"
    mcfg.generation.max_new_tokens = max_new_tokens
    d = QwenVLDescriber(mcfg)
    d.processor = tiny_processor()
    d.model = tiny_qwen35()
    return d


def train_cfg(**overrides):
    from vlmfid.train.config import compose_train

    base = ["model=qwen35", "model.quant=bf16", "train.effective_batch_size=8", "train.micro_batch_size=4",
            "train.eval_every=2", "train.save_every=1", "train.log_every=1", "data.num_workers=0",
            "data.val_limit=8", "lora.r=4", "lora.alpha=8", "train.lr=5e-3", "train.connector_lr=5e-4",
            "model.batch_size=4", "train.max_target_tokens=128"]
    return compose_train(base + [f"{k}={v}" for k, v in overrides.items()])


def write_m3di(root, n: int, split: str):
    """Mini-M3DI con captions de las 5 plantillas reales y latentes coherentes."""
    import polars as pl
    from PIL import Image

    from vlmfid.data import build_split

    shapes = ["teapot", "hare", "dragon", "cow", "armadillo", "horse", "head"]
    xs, ys = ["left", "center", "right"], ["top", "mid", "bottom"]
    names = ["tab:blue", "tab:red", "xkcd:bright yellow green", "gold"]
    hues = [0.62, 0.0, 0.25, 0.14]
    tpl = [
        'A {s} of "{c}" color is visible, positioned at the {y}-{x} of the image.',
        'A "{c}" {s} is at the {y}-{x} of the image.',
        'The {y}-{x} of the image shows a "{c}" colored {s}.',
        'At the {y}-{x} of the picture is a {s} in "{c}" color.',
        'At the {y}-{x} of the image, there is a "{c}" object in the form of a {s}.',
    ]
    d = root / split
    (d / "images").mkdir(parents=True)
    (d / "text").mkdir()
    img, txt, caps = {k: [] for k in ["object_shape", "object_xpos", "object_ypos", "object_zpos", "object_color"]}, \
        {k: [] for k in ["object_shape", "object_xpos", "object_ypos", "object_zpos", "object_color_name",
                         "object_color_index", "text_phrasing"]}, []
    for i in range(n):
        s, x, y, c, p = i % 7, i % 3, (i // 3) % 3, i % 4, i % 5
        Image.new("RGB", (64, 64), (40 * c, 20 * x, 20 * y)).save(d / "images" / f"{i:06d}.png")
        for k, val in zip(img, [s, x, y, 0, hues[c]]):
            img[k].append(val)
        for k, val in zip(txt, [s, x, y, 0, names[c], c, p]):
            txt[k].append(val)
        caps.append(tpl[p].format(s=shapes[s], c=names[c], x=xs[x], y=ys[y]))
    pl.DataFrame(img).write_csv(d / "latents_image.csv")
    pl.DataFrame(txt).write_csv(d / "latents_text.csv")
    (d / "text" / "text_raw.txt").write_text("\n".join(caps) + "\n")
    return build_split(root, split, verbose=False)


class TensorCharTokenizer(CharTokenizer):
    """Interfaz de tokenizador de HF que usa VLJEPADescriber._tok (relleno a max_length, tensores)."""

    def __call__(self, text, max_length=None, padding=None, truncation=False, return_tensors=None,
                 add_special_tokens=True, **_):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t) for t in texts]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        if return_tensors != "pt":
            return {"input_ids": ids if not isinstance(text, str) else ids[0]}
        L = max_length if padding == "max_length" else max(map(len, ids))
        att = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
        ids = [x + [self.pad_token_id] * (L - len(x)) for x in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}


def tiny_vljepa_describer(tmp, records_by_split):
    """VLJEPADescriber con un OpenVLJEPA diminuto (V-JEPA 2 + Llama + Gemma3 aleatorios, guardados en
    `tmp`). `records_by_split` alimenta el banco de recuperación en lugar del manifiesto real."""
    import sys

    import transformers as T
    from torchvision import transforms as TV

    from vlmfid.config import _load_model
    from vlmfid.models import vljepa as vj
    from vlmfid.paths import ROOT

    sys.path.insert(0, str(ROOT / "external" / "open-vljepa"))
    from openvljepa.models.vljepa import OpenVLJEPA

    torch.manual_seed(0)
    vc = T.VJEPA2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, crop_size=32, patch_size=16,
                        tubelet_size=2, frames_per_clip=8, pred_hidden_size=16, pred_num_hidden_layers=1,
                        pred_num_attention_heads=2, pred_num_mask_tokens=2)
    T.VJEPA2Model(vc).save_pretrained(tmp / "vjepa")
    T.LlamaForCausalLM(T.LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                     num_attention_heads=4, num_key_value_heads=2, vocab_size=110)
                       ).save_pretrained(tmp / "llama")
    T.Gemma3TextModel(T.Gemma3TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                         vocab_size=110)).save_pretrained(tmp / "gemma")
    m = OpenVLJEPA({"model_name": str(tmp / "vjepa"), "frozen": True, "output_dim": None},
                   {"model_name": str(tmp / "gemma"), "proj_dim": 24},
                   {"llama_name": str(tmp / "llama"), "n_keep_layers": 2, "proj_dim": 24},
                   torch_dtype=torch.bfloat16).to(torch.bfloat16).eval()

    mcfg = _load_model("vljepa")
    mcfg.quant = "bf16"
    mcfg.batch_size = 4
    mcfg.bank.max_captions = 1000
    d = vj.VLJEPADescriber(mcfg)
    d.model = m
    d.ckpt_path = tmp / "best.pt"
    d.ckpt_path.write_bytes(b"x")
    d.max_query_len, d.max_caption_len, d.num_frames = 16, 64, 8
    d.query_tok = d.target_tok = TensorCharTokenizer()
    d.transform = TV.Compose([TV.Resize(32, antialias=True), TV.CenterCrop(32), TV.ToTensor()])
    orig = vj.load_manifest
    vj.load_manifest = lambda split, **k: records_by_split[split]
    old_cache = vj.CACHE
    vj.CACHE = tmp / "cache"
    try:
        d._build_bank()
    finally:
        vj.load_manifest, vj.CACHE = orig, old_cache
    return d
