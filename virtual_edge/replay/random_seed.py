import random
from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class SeedBundle:

    seed: int
    _generator: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._generator = np.random.default_rng(self.seed)

    @property
    def generator(self) -> np.random.Generator:
        return self._generator

def seed_everything(seed: int = 42) -> np.random.Generator:
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    return SeedBundle(seed).generator

__all__ = ["SeedBundle", "seed_everything"]