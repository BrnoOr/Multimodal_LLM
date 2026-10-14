"""Orden de los datos de entrenamiento: determinista, reanudable y común a los 4 modelos.

Cada época usa una permutación de train fijada por (seed, época). El paso de optimización s de una
época consume las posiciones [s·B, (s+1)·B) de esa permutación, con B = lote efectivo. Como B es igual
en los 4 modelos, **todos ven las mismas imágenes en el mismo orden y agrupadas en los mismos lotes**;
el micro-lote solo decide cuántas de esas B pasan juntas por la GPU.

Reanudar en el paso s es saltar a la posición s·B: no hace falta guardar el estado del DataLoader.
"""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


def steps_per_epoch(n: int, effective_batch: int, drop_last: bool = True) -> int:
    return n // effective_batch if drop_last else -(-n // effective_batch)


def epoch_permutation(n: int, seed: int, epoch: int) -> list[int]:
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch))
    return torch.randperm(n, generator=g).tolist()


class EpochSampler(Sampler[int]):
    """Índices de una época a partir del paso `start_step` (inclusive)."""

    def __init__(self, n: int, effective_batch: int, seed: int, epoch: int, start_step: int = 0,
                 drop_last: bool = True):
        self.perm = epoch_permutation(n, seed, epoch)
        self.n_steps = steps_per_epoch(n, effective_batch, drop_last)
        self.start = min(start_step, self.n_steps) * effective_batch
        self.stop = min(self.n_steps * effective_batch, n)

    def __iter__(self):
        return iter(self.perm[self.start:self.stop])

    def __len__(self) -> int:
        return max(0, self.stop - self.start)
