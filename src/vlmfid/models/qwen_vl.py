"""Familia Qwen-VL (Qwen2.5-VL, Qwen3.5): resolución dinámica.

El número de tokens visuales depende de la resolución de entrada: la imagen se reescala a un
múltiplo de patch_size·merge_size (32 px en Qwen3.5) con un área total en [min_pixels, max_pixels],
y cada bloque de 32×32 px se vuelve un token. Una imagen de M3DI (224×224) da 7×7 = 49 tokens si
el `min_pixels` del checkpoint no la agranda (si su mínimo es 256², sube a 8×8 = 64). Subir
`min_pixels` aumenta los tokens visuales: es una variable experimental, no un detalle técnico, y
debe fijarse igual en todos los runs de un mismo modelo. `info()` registra el tamaño efectivo.
"""

from __future__ import annotations

from .hf_generative import HFGenerativeDescriber


class QwenVLDescriber(HFGenerativeDescriber):
    family = "qwen_vl"

    def configure_processor(self) -> None:
        super().configure_processor()
        lo, hi = self.cfg.get("min_pixels"), self.cfg.get("max_pixels")
        if lo is None and hi is None:
            return
        ip = self.processor.image_processor
        raw = getattr(ip, "size", None)
        try:  # dict o SizeDict según la versión
            size = dict(raw) if raw else {}
        except TypeError:
            size = {k: getattr(raw, k) for k in ("shortest_edge", "longest_edge") if getattr(raw, k, None)}
        # el procesador exige ambos límites (shortest_edge = min_pixels, longest_edge = max_pixels)
        if lo is not None:
            size["shortest_edge"] = int(lo)
        if hi is not None:
            size["longest_edge"] = int(hi)
        ip.size = size

    def info(self) -> dict:
        out = super().info()
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            out.update({"patch_size": getattr(ip, "patch_size", None), "merge_size": getattr(ip, "merge_size", None),
                        "size": dict(getattr(ip, "size", {}) or {})})
        return out
