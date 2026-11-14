"""Y-Decoder de VL-JEPA: lee un embedding del espacio compartido y escribe el caption.

DISEÑO PROPIO. El paper de VL-JEPA (Chen et al., 2025) solo dice que el Y-Decoder es ligero, que no
participa del entrenamiento principal y que traduce el embedding predicho a texto en inferencia (§2);
no da arquitectura ni entrenamiento. Lo que sigue respeta esas tres propiedades y las decisiones de
docs/metodologia.md §2.4:

    embedding (1536 D)  ->  normalización L2  ->  proyección lineal entrenable  ->  k vectores de prefijo
    [prefijo, BOS, caption]  ->  Llama-3.2-1B preentrenado (NF4 + LoRA)  ->  caption

  * Prefijo sobre un LM preentrenado, al estilo ClipCap (Mokady et al., 2021).
  * El LM es Llama-3.2-1B ORIGINAL, no las capas ajustadas del Predictor: el decodificador debe ser
    independiente del Predictor para que su error no se confunda con el de la predicción.
  * Cuantización y adaptación como el resto del proyecto: base en NF4 (cómputo bf16), LoRA en todas
    las capas lineales de los bloques (QLoRA §4) y la proyección de prefijo en precisión completa.
  * Solo ve el embedding: nunca la imagen ni la consulta.

Normalización de la entrada. InfoNCE compara embeddings normalizados, así que la norma de Ŝ_Y no está
supervisada y puede diferir de la de S_Y. El decodificador se entrena con S_Y y se aplica a Ŝ_Y: sin
normalizar, una diferencia de norma lo sacaría de distribución sin que cambie la información del
embedding. Por eso la entrada se lleva a norma 1 (`normalize_input: true`).

Entrenamiento (vlmfid.train.ydecoder): solo texto, pares ⟨S_Y(caption), caption⟩ de train, entropía
cruzada sobre los tokens del caption y el fin de secuencia.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .quant import hf_quantization_config, quantization_report

IGNORE = -100
LLAMA32_PAD = "<|finetune_right_pad_id|>"
# todas las capas lineales de los bloques del LM (atención y MLP); lm_head queda fuera
LORA_REGEX = r".*\.layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
PREFIX_FILE, CONFIG_FILE, ADAPTER_DIR = "prefix.pt", "decoder.yaml", "adapter"
REQUIRED = ("llama_name", "quant", "embed_dim", "prefix_len", "normalize_input", "lora.r", "lora.alpha",
            "lora.dropout")


class YDecoder:
    """Prefijo + LM. `cfg` es la sección `decoder` de la configuración (configs/ydecoder.yaml)."""

    def __init__(self, cfg: DictConfig):
        missing = [k for k in REQUIRED if OmegaConf.select(cfg, k) is None]
        if missing:
            raise ValueError(f"Faltan valores en la config del Y-Decoder (no se asumen por defecto): {missing}")
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm = None
        self.prefix: nn.Linear | None = None
        self.tok = None

    # ---------------------------------------------------------------- carga
    def load(self, trainable: bool = False, adapter_dir: Path | None = None) -> "YDecoder":
        """Construye el decodificador. `trainable=True` adjunta un LoRA nuevo; `adapter_dir` carga uno ya
        entrenado. Exactamente una de las dos opciones."""
        if trainable == (adapter_dir is not None):
            raise ValueError("load(): usa trainable=True (entrenar) o adapter_dir (inferir), no ambos ni ninguno")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = self.cfg
        self.tok = AutoTokenizer.from_pretrained(c.llama_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = LLAMA32_PAD if LLAMA32_PAD in self.tok.get_vocab() else self.tok.eos_token
        self.tok.padding_side = "right"
        if self.tok.bos_token_id is None or self.tok.eos_token_id is None:
            raise ValueError(f"{c.llama_name}: el tokenizador no define BOS/EOS")

        lm = AutoModelForCausalLM.from_pretrained(
            c.llama_name, dtype=torch.bfloat16,
            quantization_config=hf_quantization_config(str(c.quant), ["lm_head"]),
            device_map={"": 0} if torch.cuda.is_available() else None, low_cpu_mem_usage=True,
        )
        lm.requires_grad_(False)
        self.quant_report = quantization_report(lm)
        if str(c.quant) == "nf4" and self.quant_report["quantized_linears"] == 0:
            raise RuntimeError("quant=nf4 pero el LM del Y-Decoder no tiene capas cuantizadas")
        self.hidden = int(lm.config.hidden_size)
        self.prefix = nn.Linear(int(c.embed_dim), int(c.prefix_len) * self.hidden)      # fp32

        if trainable:
            from peft import LoraConfig, get_peft_model

            lm = get_peft_model(lm, LoraConfig(
                r=int(c.lora.r), lora_alpha=int(c.lora.alpha), lora_dropout=float(c.lora.dropout),
                bias="none", target_modules=LORA_REGEX, task_type="CAUSAL_LM"))
            for p in lm.parameters():                 # pesos maestros en fp32; el forward corre en autocast bf16
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
        else:
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, str(Path(adapter_dir) / ADAPTER_DIR))
            self.prefix.load_state_dict(torch.load(Path(adapter_dir) / PREFIX_FILE, map_location="cpu"))
            lm.eval()
        self.lm = lm
        self.prefix.to(self.device)
        return self

    @classmethod
    def from_dir(cls, path: str | Path) -> "YDecoder":
        """Decodificador entrenado, listo para `decode`."""
        path = Path(path)
        cfg = OmegaConf.load(path / CONFIG_FILE)
        return cls(cfg).load(adapter_dir=path)

    def save(self, path: Path, meta: dict | None = None) -> None:
        """Guarda solo el adaptador LoRA, la proyección de prefijo y la config (nunca el LM completo)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(str(path / ADAPTER_DIR))
        torch.save(self.prefix.state_dict(), path / PREFIX_FILE)
        cfg = OmegaConf.merge(self.cfg, OmegaConf.create({"meta": meta or {}}))
        (path / CONFIG_FILE).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    @property
    def meta(self) -> dict:
        return OmegaConf.to_container(self.cfg.get("meta") or OmegaConf.create({}), resolve=True)

    # ------------------------------------------------------------- parámetros
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.lm.parameters() if p.requires_grad] + list(self.prefix.parameters())

    def count_trainable(self) -> dict:
        """Conteo de entrenables por grupo; falla si aparece algo que no sea LoRA."""
        n_lora, other = 0, []
        for name, p in self.lm.named_parameters():
            if p.requires_grad:
                if "lora_" in name:
                    n_lora += p.numel()
                else:
                    other.append(name)
        if other:
            raise RuntimeError(f"Parámetros entrenables inesperados en el LM del Y-Decoder: {other[:5]}")
        if n_lora == 0:
            raise RuntimeError("LoRA no se aplicó a ninguna capa del Y-Decoder")
        n_prefix = sum(p.numel() for p in self.prefix.parameters())
        return {"trainable_lora": n_lora, "trainable_prefix": n_prefix, "trainable_total": n_lora + n_prefix}

    # ---------------------------------------------------------------- forward
    def _prefix_embeds(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, k, H): los k vectores de prefijo en el espacio de entrada del LM."""
        if emb.ndim != 2 or emb.shape[1] != int(self.cfg.embed_dim):
            raise ValueError(f"Se esperaba (B, {self.cfg.embed_dim}) y llegó {tuple(emb.shape)}")
        e = emb.to(self.device, dtype=torch.float32)
        if bool(self.cfg.normalize_input):
            e = F.normalize(e, dim=-1)
        return self.prefix(e).view(e.shape[0], int(self.cfg.prefix_len), self.hidden)

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    def loss(self, emb: torch.Tensor, captions: list[str]) -> tuple[torch.Tensor, int]:
        """Entropía cruzada media por token del caption (y su EOS). Devuelve (pérdida, n.º de tokens)."""
        if len(captions) != emb.shape[0]:
            raise ValueError(f"{emb.shape[0]} embeddings para {len(captions)} captions")
        tok, k = self.tok, int(self.cfg.prefix_len)
        ids = [tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for c in captions]
        L, B = max(len(x) for x in ids), len(ids)
        tgt = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, x in enumerate(ids):
            tgt[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        tgt, mask = tgt.to(self.device), mask.to(self.device)
        bos = torch.full((B, 1), tok.bos_token_id, dtype=torch.long, device=self.device)

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            tok_emb = self._embed_tokens(torch.cat([bos, tgt], dim=1))                 # (B, 1+L, H)
            inputs = torch.cat([self._prefix_embeds(emb).to(tok_emb.dtype), tok_emb], dim=1)   # (B, k+1+L, H)
            attn = torch.cat([torch.ones((B, k + 1), dtype=torch.long, device=self.device), mask], dim=1)
            logits = self.lm(inputs_embeds=inputs, attention_mask=attn, use_cache=False).logits
        # la posición k (BOS) predice el primer token del caption; la k+L-1 predice el último (EOS)
        pred = logits[:, k:k + L].float()
        labels = tgt.masked_fill(mask == 0, IGNORE)
        n = int(mask.sum())
        if pred.shape[:2] != labels.shape:
            raise RuntimeError(f"Desalineación: logits {tuple(pred.shape[:2])} vs etiquetas {tuple(labels.shape)}")
        return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), labels.reshape(-1), ignore_index=IGNORE), n

    @torch.inference_mode()
    def decode(self, emb: torch.Tensor, batch_size: int = 64) -> list[str]:
        """Caption generado para cada embedding, con decodificación determinista (greedy)."""
        g = OmegaConf.to_container(self.cfg.get("generation") or OmegaConf.create({}), resolve=True)
        if g.get("max_new_tokens") is None:
            raise ValueError("Falta decoder.generation.max_new_tokens")
        out: list[str] = []
        was_training = self.lm.training
        self.lm.eval()
        for i in range(0, emb.shape[0], batch_size):
            e = emb[i:i + batch_size]
            B = e.shape[0]
            bos = torch.full((B, 1), self.tok.bos_token_id, dtype=torch.long, device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                bos_emb = self._embed_tokens(bos)
                inputs = torch.cat([self._prefix_embeds(e).to(bos_emb.dtype), bos_emb], dim=1)
                attn = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
                # todas las filas tienen el mismo largo de entrada: no hace falta relleno
                gen = self.lm.generate(inputs_embeds=inputs, attention_mask=attn, do_sample=False, num_beams=1,
                                       max_new_tokens=int(g["max_new_tokens"]),
                                       eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
            # con `inputs_embeds`, generate devuelve solo los tokens nuevos
            out += [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        if was_training:
            self.lm.train()
        return out
