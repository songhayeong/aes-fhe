import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Any
from aes_xor_fhe.engine_context import EngineContext


class MultivariateCache:
    def __init__(self, path: Path):
        self.path = path
        self._plain_cache: Dict[int, Dict[Tuple[int,int], Any]] = {}

    @lru_cache(maxsize=1)
    def load_coeffs(self) -> Dict[Tuple[int,int], complex]:
        """
        2D‐LUT 계수 JSON을 읽어서 (i,j)->complex 계수 dict로 반환.
        entries 항목은 [i, j, real, imag] 형태여야 합니다.
        """
        data = json.loads(self.path.read_text(encoding='utf-8'))
        coeffs: Dict[Tuple[int,int], complex] = {}
        for entry in data["entries"]:
            if len(entry) != 4:
                raise ValueError(f"expected 4‐tuple in multivariate LUT, got: {entry}")
            i, j, re, im = entry
            coeffs[(int(i), int(j))] = complex(re, im)
        return coeffs



    def get_plaintext_coeffs(self, ctx: EngineContext) -> Dict[Tuple[int,int], Any]:
        sc = ctx.engine.slot_count
        if sc in self._plain_cache:
            return self._plain_cache[sc]
        coeffs = self.load_coeffs()
        pt_cache: Dict[Tuple[int,int], Any] = {}
        for (i,j), c in coeffs.items():
            vec = np.full(sc, c, dtype=np.complex128)
            pt_cache[(i,j)] = ctx.engine.encode(vec)
        self._plain_cache[sc] = pt_cache
        return pt_cache