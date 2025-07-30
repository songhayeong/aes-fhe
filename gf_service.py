# aes_xor_fhe/gf_service.py
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from desilofhe import Ciphertext, Plaintext

from aes_xor_fhe.xor_service import EngineWrapper, CoefficientCache, XORService, ZetaEncoder


class GFService:
    """
    Homomorphic GF(2^8) multiplication by 2 and 3, via 4-to-4 2D LUTs.
    """

    def __init__(self, eng_wrap: EngineWrapper,
                 xor_svc: XORService,
                 gf2_path: Path,
                 gf3_path: Path):
        self.eng = eng_wrap
        self.xor_svc = xor_svc
        # 2D LUT coeff caches: keys are (hi, lo)
        self.gf2_cache = CoefficientCache(gf2_path)
        self.gf3_cache = CoefficientCache(gf3_path)

    def _eval_4to4(self, enc_vec: Any, cache: CoefficientCache) -> Any:
        # split to hi/lo 4-bit
        hi, lo = self.xor_svc.extract_nibbles(enc_vec)
        eng = self.eng
        # bulid power bases for hi and lo
        bx = eng.make_power_basis(hi, 16)
        by = eng.make_power_basis(lo, 16)
        # get coefficients: dict of (i, j) -> Plaintext
        pts = cache.get_plaintext_coeffs(eng)
        # accumulate
        res = eng.multiply(enc_vec, 0.0)
        for (i, j), pt in pts.items():
            if i == 0 and j == 0:
                term = pt
            elif i == 0:
                term = eng.multiply(by[j], pt)
            elif j == 0:
                term = eng.multiply(bx[i], pt)
            else:
                t = eng.multiply(bx[i], by[j], eng.relin_key)
                term = eng.multiply(t, pt, eng.relin_key)
            res = eng.add(res, term)
        return res

    def _eval_2d_lut(self, ct_hi: Ciphertext, ct_lo: Ciphertext, cache: CoefficientCache) -> Ciphertext:
        eng = self.eng
        pts: Dict[Tuple[int, int], Plaintext] = cache.get_plaintext_coeffs(eng)
        n = 16  # should be 16
        half = n // 2  # = 8

        # what (i,j) we actually need
        needed = set(pts.keys()) - {(0, 0)}
        max_i = max(i for i, _ in needed) if needed else 1
        max_j = max(j for _, j in needed) if needed else 1
        deg_i = min(max_i, half)
        deg_j = min(max_j, half)

        # build x^1…x^deg_i  and y^1…y^deg_j
        pow_hi = eng.make_power_basis(ct_hi, deg_i)  # length=deg_i
        pow_lo = eng.make_power_basis(ct_lo, deg_j)

        # reconstruct any higher exponents via conjugation symmetry
        basis_hi: Dict[int, Ciphertext] = {0: eng.add_plain(ct_hi, 1.0)}
        for i in range(1, n):
            if i <= deg_i:
                basis_hi[i] = pow_hi[i - 1]
            else:
                basis_hi[i] = eng.conjugate(pow_hi[n - i - 1])

        basis_lo: Dict[int, Ciphertext] = {0: eng.add_plain(ct_lo, 1.0)}
        for j in range(1, n):
            if j <= deg_j:
                basis_lo[j] = pow_lo[j - 1]
            else:
                basis_lo[j] = eng.conjugate(pow_lo[n - j - 1])

        # now accumulate ∑_{i,j} c_{i,j} * (hi^i)*(lo^j)
        out = eng.multiply(ct_hi, 0.0)  # zero ciphertext
        for (i, j), pt in pts.items():
            if i == 0 and j == 0:
                term = pt
            elif i == 0:
                term = eng.multiply(basis_lo[j], pt)
            elif j == 0:
                term = eng.multiply(basis_hi[i], pt)
            else:
                # hi^i * lo^j
                tmp = eng.multiply(basis_hi[i], basis_lo[j], eng.relin_key)
                term = eng.multiply(tmp, pt, eng.relin_key)
            out = eng.add(out, term)
        return out

    def mul1(self, ct):
        # multiply by 1 -> identity
        return ct

    # def mul2(self, enc_vec: Any) -> Any:
    #     """
    #     GF(2^8) * 2
    #     """
    #     return self._eval_4to4(enc_vec, self.gf2_cache)
    #
    # def mul3(self, enc_vec: Any) -> Any:
    #     """
    #     use separate coeff table for x3
    #     """
    #     return self._eval_4to4(enc_vec, self.gf3_cache)

    def mul2(self, ct: Ciphertext) -> Ciphertext:
        return self._eval_2d_lut(ct_hi=ct, ct_lo=ct, cache=self.gf2_cache)

    def mul3(self, ct: Ciphertext) -> Ciphertext:
        return self._eval_2d_lut(ct_hi=ct, ct_lo=ct, cache=self.gf3_cache)
