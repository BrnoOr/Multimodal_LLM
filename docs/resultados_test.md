# Resultados en test (reporte final, una sola vez)

> Lo medido en test con el plan congelado (`runs/final_test/plan.lock.yaml`, 14 condiciones; `configs/final_test.yaml`).
> Test se usó **una sola vez** (`scripts/final_test.py --confirm`, job `final_test`, 24-11-2026), después de cerrar
> todas las decisiones en val (`resultados_etapa1.md`, `resultados_etapa2.md`). Ninguna decisión se tomó ni se
> tomará con estas cifras. Fuente: `runs/final_test/tabla_final.csv` (cluster) → `reports/tables/final/`.
> Comparación válida: entre nuestros 4 modelos bajo el mismo protocolo, no contra cifras de los papers (modelos de
> 7B o más) [CLAUDE.md §2].

## 1. Tabla final (10 000 imágenes de test por condición)

Las 14 corridas terminaron (`complete = True`) con n = 10 000.

| Modelo | Condición | Prompt | Precisión | forma | x | y | color | **attr_mean** | Bleu-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | zero-shot | `p1_attributes` | NF4 | 0.144 | 0.819 | 0.577 | 0.124 | 0.416 | 0.000 | 0.158 | 0.000 |
| | | | bf16 | 0.180 | 0.789 | 0.673 | 0.127 | 0.442 | 0.003 | 0.153 | 0.000 |
| | ajustado | `p1_attributes` | NF4 | 0.999 | 1.000 | 0.999 | 0.273 | **0.818** | 0.467 | 0.558 | 2.502 |
| | | | bf16 | 0.993 | 1.000 | 0.981 | 0.305 | 0.819 | 0.421 | 0.545 | 2.764 |
| InternVL3-1B | zero-shot | `p1_attributes` | NF4 | 0.202 | 0.387 | 0.274 | 0.214 | 0.269 | 0.013 | 0.226 | 0.000 |
| | | | bf16 | 0.224 | 0.292 | 0.131 | 0.219 | 0.216 | 0.004 | 0.226 | 0.000 |
| | ajustado | `p1_attributes` | NF4 | 0.997 | 1.000 | 0.988 | 0.282 | **0.817** | 0.422 | 0.544 | 2.741 |
| | | | bf16 | 1.000 | 1.000 | 1.000 | 0.250 | 0.813 | 0.466 | 0.557 | 2.443 |
| LLaVA-OV-0.5B | zero-shot | `p3_json` | NF4 | 0.375 | 0.344 | 0.333 | 0.193 | 0.311 | 0.000 | 0.077 | 0.060 |
| | | | bf16 | 0.389 | 0.337 | 0.349 | 0.211 | 0.322 | 0.000 | 0.078 | 0.065 |
| | ajustado | `p1_attributes` | NF4 | 1.000 | 1.000 | 1.000 | 0.267 | **0.817** | 0.461 | 0.554 | 2.482 |
| | | | bf16 | 0.999 | 1.000 | 0.994 | 0.310 | 0.826 | 0.410 | 0.561 | 2.483 |
| VL-JEPA | zero-shot | `p4m_multiexample` | NF4 | 0.004 | 0.000 | 0.000 | 0.013 | 0.004 | 0.000 | 0.162 | 0.009 |
| | ajustado | `p1_attributes` | NF4 | 0.999 | 0.997 | 1.000 | 0.240 | **0.809** | 0.434 | 0.551 | 2.343 |

Cifras con 4 decimales en `tabla_final.csv`. VL-JEPA no tiene condiciones bf16: el zero-shot se canceló en CE1 y el
ajustado en P10′ (su control bf16 se reporta solo en el espacio de embeddings, `resultados_etapa2.md` §5).

Omisiones: 0 en las 8 condiciones ajustadas. En zero-shot se repiten las de val: InternVL3 y Qwen3.5 omiten la forma
en el 73–78 % y VL-JEPA omite casi todo (forma 0.99, x e y ≈ 1.00, color 0.60) con el lector genérico.

## 2. Test frente a val

`attr_mean` en val (`resultados_etapa1.md` §4b, `resultados_etapa2.md` §6–7) y en test, mismas condiciones.

| Modelo | Condición | Precisión | val | test | test − val |
|---|---|---|---|---|---|
| Qwen3.5 | ajustado | NF4 | 0.8159 | 0.8177 | +0.0018 |
| InternVL3 | ajustado | NF4 | 0.8152 | 0.8166 | +0.0014 |
| LLaVA-OV | ajustado | NF4 | 0.8147 | 0.8165 | +0.0018 |
| VL-JEPA | ajustado | NF4 | 0.8073 | 0.8091 | +0.0018 |
| Qwen3.5 | ajustado | bf16 | 0.8186 | 0.8195 | +0.0009 |
| InternVL3 | ajustado | bf16 | 0.8120 | 0.8126 | +0.0006 |
| LLaVA-OV | ajustado | bf16 | 0.8259 | 0.8257 | −0.0002 |
| Qwen3.5 | zero-shot | NF4 | 0.416 | 0.416 | 0.000 |
| LLaVA-OV | zero-shot | NF4 | 0.309 | 0.311 | +0.002 |
| InternVL3 | zero-shot | NF4 | 0.274 | 0.269 | −0.005 |
| VL-JEPA | zero-shot | NF4 | 0.0039 | 0.0043 | +0.0004 |

- **Test reproduce val.** Las 14 condiciones quedan a ≤ 0.005 de su cifra en val, y el orden se mantiene en todas
  las comparaciones: en NF4 ajustado, Qwen3.5 > InternVL3 > LLaVA-OV > VL-JEPA, igual que en val.
- Los cuatro ajustados NF4 suben lo mismo (+0.0014 a +0.0018): es una diferencia del split (algo más de acierto en
  color en test), no de un modelo. No hay señal de sobreajuste a val en la selección de checkpoint ni de prompt.

## 3. Lectura

**Etapa 1 (zero-shot).** Se confirma en test lo de val: los VLM superan a la clase mayoritaria (0.222), con margen
desigual (Qwen3.5 0.416, LLaVA-OV 0.311, InternVL3 0.269) y VL-JEPA, con el lector genérico, queda en 0.004.

**Etapa 2 (ajuste).**

- **El ajuste iguala a los tres VLM:** 0.8165–0.8177 en NF4, a 0.0012 entre sí. Forma y posición quedan resueltas
  (≥ 0.988) y todo el error que queda es de color (0.267–0.282), con techo 0.410 por la paleta que la imagen no
  muestra (techo de `attr_mean` ≈ 0.853).
- **VL-JEPA ajustado llega a 0.809**, a 0.007–0.009 de los VLM. Forma y posición también quedan resueltas
  (≥ 0.997); la diferencia está otra vez en el color (0.240 frente a 0.267–0.282).
- **bf16 frente a NF4** (ajustado): LLaVA-OV +0.009, Qwen3.5 +0.002, InternVL3 −0.004; la diferencia está en el color.
  Mismo signo y tamaño que en val.
- **Métricas de texto:** CIDEr 2.34–2.76 en los ajustados; no ordenan igual que la fidelidad (InternVL3 NF4 tiene el
  CIDEr más alto con la misma `attr_mean` que LLaVA-OV), como en val (`resultados_etapa2.md` §6).

**Incertidumbre (aproximación, no es la prueba pareada).** Con n = 10 000 y errores independientes entre imágenes,
el error estándar binomial es ≈ 0.0045 en el color y ≈ 0.0011 en `attr_mean` (dominado por el color: forma y
posición ≈ 1). Una diferencia entre dos condiciones, tratada como no pareada, tiene error estándar ≈ 0.0016 en
`attr_mean`. Con eso:

- Las diferencias entre los tres VLM ajustados (≤ 0.0012) **no se distinguen del ruido**.
- La brecha de VL-JEPA (0.007–0.009) es de 4.5 a 5.5 errores estándar, y en color (0.027–0.041) de 4 a 6.5.
- La aproximación no pareada es conservadora si los aciertos de dos modelos sobre la misma imagen están
  correlacionados positivamente. Las pruebas pareadas (McNemar, bootstrap) se hicieron en val (notebook 03); en
  test no se han corrido (ver §4).

## 4. Pendiente

- **Pruebas pareadas en test** (McNemar por atributo, bootstrap de `attr_mean`) sobre las predicciones de
  `runs/final_*`: decisión del usuario. Leen solo las salidas ya generadas, no cambian ninguna decisión y no vuelven a
  correr los modelos; pero hoy solo `scripts/final_test.py` toca test (CLAUDE.md §5), así que habría que agregarlas
  ahí o autorizar una excepción explícita para el análisis.
- Copiar la tabla a `reports/tables/final/` (`scripts/collect_reports.py` en el cluster → scp → commit en el laptop).
