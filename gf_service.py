# aes_xor_fhe/gf_service.py
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
from desilofhe import Ciphertext, Plaintext

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.xor_service import EngineWrapper, CoefficientCache, XORService, ZetaEncoder


def _load_coeffs(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    n = data["n"]
    coeffs = np.zeros(n, dtype=np.complex128)
    for i, re, im in data["entries"]:
        coeffs[int(i)] = re + 1j * im
    return coeffs


class GFService:
    """
    GF(2^8) ×2, ×3, ×9, ×11, ×13, ×14 동형암호 연산 서비스
    (이 예제에선 ×2, ×3만 full-256 LUT 다항식으로 처리)
    """

    def __init__(
            self,
            eng_wrap: EngineWrapper,
            xor_svc: XORService,
    ):
        self.eng = eng_wrap
        self.xor_svc = xor_svc

        base = Path(__file__).parent / "generator/coeffs"

        # 256-entry full 8→8 LUT 다항식 계수 로드
        self.coeffs2_hi = _load_coeffs(base / "gf2_hi_coeffs.json")
        self.coeffs2_lo = _load_coeffs(base / "gf2_lo_coeffs.json")
        self.coeffs3_hi = _load_coeffs(base / "gf3_hi_coeffs.json")
        self.coeffs3_lo = _load_coeffs(base / "gf3_lo_coeffs.json")

        sc = self.eng.engine.slot_count
        # encode to plaintexts
        self.pt2_hi = [self.eng.encode(np.full(sc, c, dtype=np.complex128))
                       for c in self.coeffs2_hi]
        self.pt2_lo = [self.eng.encode(np.full(sc, c, dtype=np.complex128))
                       for c in self.coeffs2_lo]
        self.pt3_hi = [self.eng.encode(np.full(sc, c, dtype=np.complex128))
                       for c in self.coeffs3_hi]
        self.pt3_lo = [self.eng.encode(np.full(sc, c, dtype=np.complex128))
                       for c in self.coeffs3_lo]

    def _eval_1d_lut(self, ct, pt_list):
        powers = self.eng.make_power_basis(ct, len(pt_list) - 1)
        out = self.eng.multiply(ct, 0.0)
        out = self.eng.add(out, pt_list[0])
        for i, pt in enumerate(pt_list[1:], start=1):
            out = self.eng.add(
                out,
                self.eng.multiply(powers[i - 1], pt, self.eng.relin_key)
            )
        return out

    def mul1(self, ct):
        # identity
        return ct

    def mul2(self, ct):
        hi = self._eval_1d_lut(ct, self.pt2_hi)
        lo = self._eval_1d_lut(ct, self.pt2_lo)
        return hi, lo

    def mul3(self, ct):
        hi = self._eval_1d_lut(ct, self.pt3_hi)
        lo = self._eval_1d_lut(ct, self.pt3_lo)
        return hi, lo

    # def _eval_1d_lut(self, ct: Any, cache: CoefficientCache) -> Any:
    #     """
    #     1차원 LUT 다항식 평가
    #     cache.get_plaintext_coeffs() → dict[k] = Plaintext(coef_k)
    #     """
    #     eng = self.eng_wrap
    #     pts: Dict[int, Any] = cache.get_plaintext_coeffs(eng)
    #     # 필요한 최대 차수
    #     max_k = max(pts.keys())
    #     # power basis: [ct^1, …, ct^max_k]
    #     basis = eng.make_power_basis(ct, max_k)
    #     # 결과 초기화
    #     res = eng.multiply(ct, 0.0)
    #     for k, pt in pts.items():
    #         if k == 0:
    #             term = pt
    #         else:
    #             # basis[k-1] == ct^k
    #             term = eng.multiply(basis[k - 1], pt)
    #         res = eng.add(res, term)
    #     return res
