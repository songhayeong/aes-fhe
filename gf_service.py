# aes_xor_fhe/gf_service.py
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
from desilofhe import Ciphertext, Plaintext

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.xor_service import EngineWrapper, CoefficientCache, XORService, ZetaEncoder


def load_json_coeffs(path: Path) -> np.ndarray:
    """
    Load LUT coefficients from JSON file (entries array of [i, real, imag]).
    Returns full-length numpy array of complex coefficients.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    n = data.get('n', None) or len(data['entries'])
    coeffs = np.zeros(n, dtype=np.complex128)
    for entry in data['entries']:
        i, real, imag = entry
        coeffs[int(i)] = real + 1j * imag
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
        gf2_path: Path,
        gf3_path: Path,
    ):
        self.eng_wrap = eng_wrap
        self.xor_svc  = xor_svc

        # 256-entry full 8→8 LUT 다항식 계수 로드
        self.gf2_full_cache = CoefficientCache(gf2_path)
        self.gf3_full_cache = CoefficientCache(gf3_path)

    def mul1(self, ct):
        # identity
        return ct

    def mul2(self, ct):
        # GF×2: full 8→8 LUT 다항식 평가
        return self._eval_1d_lut(ct, self.gf2_full_cache)

    def mul3(self, ct):
        # GF×3: full 8→8 LUT 다항식 평가
        return self._eval_1d_lut(ct, self.gf3_full_cache)

    def _eval_1d_lut(self, ct: Any, cache: CoefficientCache) -> Any:
        """
        1차원 LUT 다항식 평가
        cache.get_plaintext_coeffs() → dict[k] = Plaintext(coef_k)
        """
        eng = self.eng_wrap
        pts: Dict[int, Any] = cache.get_plaintext_coeffs(eng)
        # 필요한 최대 차수
        max_k = max(pts.keys())
        # power basis: [ct^1, …, ct^max_k]
        basis = eng.make_power_basis(ct, max_k)
        # 결과 초기화
        res = eng.multiply(ct, 0.0)
        for k, pt in pts.items():
            if k == 0:
                term = pt
            else:
                # basis[k-1] == ct^k
                term = eng.multiply(basis[k-1], pt)
            res = eng.add(res, term)
        return res
