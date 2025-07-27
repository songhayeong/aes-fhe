#!/usr/bin/env python3
"""
Refactored CKKS-based 4-bit XOR LUT service. & AddRoundKey LUT service
"""
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Any

import numpy as np
from desilofhe import Engine
from aes_xor_fhe.engine_context import EngineContext


class XORConfig:
    """
    Configuration parameters for the XOR service.
    """

    def __init__(
            self,
            coeffs_path: Path = Path(__file__).with_name("xor_mono_coeffs.json"),
            nibble_hi_path: Path = Path(__file__).with_name("nibble_hi_coeffs.json"),
            nibble_lo_path: Path = Path(__file__).with_name("nibble_lo_coeffs.json"),
            max_level: int = 22,
            mode: str = "parallel",
            thread_count: int = 8,
            device_id: int = 0,
    ):
        self.coeffs_path = coeffs_path
        self.nibble_hi_path = nibble_hi_path
        self.nibble_lo_path = nibble_lo_path
        self.max_level = max_level
        self.mode = mode
        self.thread_count = thread_count
        self.device_id = device_id


class EngineWrapper:
    """
    Wrapper around CKKS_EngineContext and Engine for convenience.
    """

    def __init__(self, config: XORConfig):
        ctx = EngineContext(
            signature=2,
            max_level=config.max_level,
            mode=config.mode,
            thread_count=config.thread_count,
            device_id=config.device_id,
        )
        self.ctx = ctx
        self.engine: Engine = ctx.engine
        self.public_key = ctx.public_key
        self.secret_key = ctx.secret_key
        self.relin_key = ctx.relinearization_key
        self.conj_key = ctx.conjugation_key

    def encrypt(self, data: np.ndarray):
        return self.engine.encrypt(data, self.public_key)

    def decrypt(self, ct) -> np.ndarray:
        return self.engine.decrypt(ct, self.secret_key)

    def encode(self, vec: np.ndarray):
        return self.engine.encode(vec)

    def multiply(self, a, b, relin_key=None):
        if relin_key:
            return self.engine.multiply(a, b, relin_key)
        return self.engine.multiply(a, b)

    def add(self, a, b):
        return self.engine.add(a, b)

    def add_plain(self, ct, val):
        try:
            return self.engine.add_plain(ct, val)
        except AttributeError:
            pt = self.engine.encode(np.full(self.engine.slot_count, val, dtype=np.complex128))
            return self.engine.add(ct, pt)

    def make_power_basis(self, ct, degree: int):
        return self.engine.make_power_basis(ct, degree, self.relin_key)

    def conjugate(self, ct):
        return self.engine.conjugate(ct, self.conj_key)

    def multiply_plain(self, ct, val):
        """
        Multiply ciphertext by plaintext scaler or vector efficiently.
        """
        if np.isscalar(val):
            return self.engine.multiply(ct, val)
        pt = self.engine.encode(np.array(val, dtype=np.complex128))
        return self.engine.multiply(ct, pt)


class ZetaEncoder:
    """
    Encode integers to roots of unity (zeta) and decode back.
    """

    @staticmethod
    def to_zeta(arr: np.ndarray, modulus: int = 16) -> np.ndarray:
        return np.exp(-2j * np.pi * (arr % modulus) / modulus)

    @staticmethod
    def from_zeta(z_arr: np.ndarray, modulus: int = 16) -> np.ndarray:
        angles = np.angle(z_arr)
        k = (-angles * modulus) / (2 * np.pi)
        return np.mod(np.rint(k), modulus).astype(np.uint8)


class CoefficientCache:
    """
    Load and cache FFT-derived polynomial coefficients and their plaintext encodings.
    """

    def __init__(self, path: Path):
        self.path = path
        self._plain_cache: Dict[int, Dict[Tuple[int, int], object]] = {}

    # @lru_cache(maxsize=1)
    # def load_coeffs(self) -> Dict[Tuple[int, int], complex]:
    #     with open(self.path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     return {
    #         (int(i), int(j)): complex(r, im)
    #         for i, j, r, im in data['entries']
    #         if r or im
    #     }
    @lru_cache(maxsize=1)
    def load_coeffs(self) -> Dict[Any, complex]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        coeffs: Dict[Any, complex] = {}
        for entry in data['entries']:
            # nibble JSON: [i, real, imag]
            if len(entry) == 3:
                i, real, imag = entry
                coeffs[int(i)] = real + 1j * imag
            # monomial JSON: [i, j, real, imag]
            elif len(entry) == 4:
                i, j, real, imag = entry
                coeffs[(int(i), int(j))] = real + 1j * imag
            else:
                raise ValueError(f"Unrecognized entry format: {entry}")
        return coeffs

    def get_plaintext_coeffs(self, engine_wrapper: EngineWrapper) -> Dict[Any, object]:
        sc = engine_wrapper.engine.slot_count
        if sc in self._plain_cache:
            return self._plain_cache[sc]

        coeffs = self.load_coeffs()
        pt_dict: Dict[Tuple[int, int], object] = {}
        for key, val in coeffs.items():
            vec = np.full(sc, val, dtype=np.complex128)
            pt_dict[key] = engine_wrapper.encode(vec)

        self._plain_cache[sc] = pt_dict
        return pt_dict


class XORService:
    """
    High-level API for performing 4-bit XOR on encrypted data.
    """

    def __init__(self, engine_wrapper: EngineWrapper,
                 coeff_cache: CoefficientCache,
                 nibble_hi_path: CoefficientCache,
                 nibble_lo_path: CoefficientCache):
        self.eng_wrap = engine_wrapper
        self.coeff_cache = coeff_cache
        self.nibble_hi_cache = nibble_hi_path
        self.nibble_lo_cache = nibble_lo_path

    @property
    def eng(self) -> EngineWrapper:
        """
        Shortcut alias so that tests can refer to xor_svc.eng.encrypt/decrypt/engine
        """
        return self.eng_wrap

    def _build_power_basis(self, ct) -> Dict[int, object]:
        # build t^0..t^8 and t^9..t^15 via conjugation
        eng = self.eng_wrap
        pos = eng.make_power_basis(ct, 8)
        basis = {0: eng.add_plain(ct, 1.0)}
        for i, c in enumerate(pos, 1):
            basis[i] = c
        for k in range(1, 8):
            basis[16 - k] = eng.conjugate(pos[k - 1])
        return basis

    def xor_cipher(self, enc_a, enc_b):
        eng = self.eng_wrap
        bx = self._build_power_basis(enc_a)
        by = self._build_power_basis(enc_b)
        pts = self.coeff_cache.get_plaintext_coeffs(eng)

        res = eng.multiply(enc_a, 0.0)
        for (i, j), pt in pts.items():
            term = eng.multiply(bx[i], by[j], eng.relin_key)
            res = eng.add(res, eng.multiply(term, pt))
        return res

    def xor(self, a_int: np.ndarray, b_int: np.ndarray) -> np.ndarray:
        # encode
        za = ZetaEncoder.to_zeta(a_int)
        zb = ZetaEncoder.to_zeta(b_int)
        enc_a = self.eng_wrap.encrypt(za)
        enc_b = self.eng_wrap.encrypt(zb)
        # homomorphic XOR
        res_ct = self.xor_cipher(enc_a, enc_b)
        # decrypt & decode
        dec = self.eng_wrap.decrypt(res_ct)
        return ZetaEncoder.from_zeta(dec)

    # def extract_nibbles(self, enc_vec) -> Tuple[object, object]:
    #     eng = self.eng_wrap
    #     hi_pts = self.nibble_hi_cache.get_plaintext_coeffs(eng)
    #     lo_pts = self.nibble_lo_cache.get_plaintext_coeffs(eng)
    #     powers = eng.make_power_basis(enc_vec, 256)
    #     enc_hi = eng.multiply(enc_vec, 0.0)
    #     for key, pt in hi_pts.items():
    #         i = key if isinstance(key, int) else key[0]
    #         term = pt if i == 0 else eng.multiply(powers[i - 1], pt)
    #         enc_hi = eng.add(enc_hi, term)
    #     enc_lo = eng.multiply(enc_vec, 0.0)
    #     for key, pt in lo_pts.items():
    #         i = key if isinstance(key, int) else key[0]
    #         term = pt if i == 0 else eng.multiply(powers[i - 1], pt)
    #         enc_lo = eng.add(enc_lo, term)
    #     return enc_hi, enc_lo
    # conjugate version으로 다시 만들어보자 !!
    def extract_nibbles(self, enc_vec) -> Tuple[Any, Any]:
        """
        SIMD nibble extraction using conjugate-reduced power basis.
        Returns (enc_hi, enc_lo) packing Zeta^(hi_nibble) and Zeta^(lo_nibble)
        """
        eng = self.eng_wrap
        sc = eng.engine.slot_count
        n = 256
        half = n // 2

        # 1) generate power-basis for exponents 1..128
        pos = eng.make_power_basis(enc_vec, half)  # [ct^1, ... , ct^128]

        # 2) build full dictionary of t^k for k=0...255
        basis: Dict[int, Any] = {0: eng.add_plain(enc_vec, 1.0)}  # 상수항 더해주기 위함
        for k in range(1, half + 1):
            basis[k] = pos[k - 1]
        # use conjugate to get the upper half : t^(n-k) = conj(t^k)
        for k in range(half + 1, n):
            # n-k in [127..1], so pos[(n-k)-1] exists
            basis[k] = eng.conjugate(pos[n - k - 1])

        # 3) hi-nibble LUT
        hi_pts = self.nibble_hi_cache.get_plaintext_coeffs(eng)  # key are in [0...255]
        enc_hi = eng.multiply(enc_vec, 0.0)
        for idx, pt in hi_pts.items():
            term = pt if idx == 0 else eng.multiply(basis[idx], pt)
            enc_hi = eng.add(enc_hi, term)

        # 4) lo-nibble LUT
        lo_pts = self.nibble_lo_cache.get_plaintext_coeffs(eng)
        enc_lo = eng.multiply(enc_vec, 0.0)
        for idx, pt in lo_pts.items():
            term = pt if idx == 0 else eng.multiply(basis[idx], pt)
            enc_lo = eng.add(enc_lo, term)

        return enc_hi, enc_lo

    def add_round_key(self, enc_state, round_key: np.ndarray) -> Any:
        eng = self.eng_wrap
        sc = eng.engine.slot_count

        # 1) 키를 Zeta^256로 인코딩 및 암호화
        zrk = ZetaEncoder.to_zeta(round_key, modulus=256)
        if zrk.size < sc:
            zrk = np.pad(zrk, (0, sc - zrk.size), constant_values = 1.0)
        enc_key = eng.encrypt(zrk)

        # 2) 니블 분할
        sx_hi, sx_lo = self.extract_nibbles(enc_state)
        ky_hi, ky_lo = self.extract_nibbles(enc_key)

        # 3) 4비트 동형 XOR
        x_hi = self.xor_cipher(sx_hi, ky_hi)
        x_lo = self.xor_cipher(sx_lo, ky_lo)

        # 4) 바이트 재조합 : Zeta^256^16으로 상위 니블 스케일 + 하위 니블 곱
        # z16 = np.exp(-2j*np.pi / 256) ** 16
        # pt_z16 = eng.encode(np.full(sc, z16, dtype=np.complex128))
        # hi_scaled = eng.multiply(x_hi, pt_z16)
        hi_powers = eng.make_power_basis(x_hi, 16)
        hi16 = hi_powers[15]
        return eng.multiply(hi16, x_lo, eng.relin_key)



def main():
    config = XORConfig()
    eng_wrap = EngineWrapper(config)
    coeff_cache = CoefficientCache(config.coeffs_path)
    service = XORService(eng_wrap, coeff_cache)

    np.random.seed(42)
    a_int = np.random.randint(0, 16, size=32768, dtype=np.uint8)
    b_int = np.random.randint(0, 16, size=32768, dtype=np.uint8)
    expected = a_int ^ b_int

    t0 = time.perf_counter()
    result = service.xor(a_int, b_int)
    t1 = time.perf_counter()

    print(f"Elapsed: {t1 - t0:.4f}s, Match: {np.all(result == expected)}")

    # AddRoundKey SIMD example
    ct = service.eng_wrap.encrypt(ZetaEncoder.to_zeta(a_int, modulus=256))
    ark_ct = service.add_round_key(ct, b_int[:ct.engine.slot_count])
    dec = service.eng_wrap.decrypt(ark_ct)
    print("AddRoundKey OK:",
          np.all(ZetaEncoder.from_zeta(dec, modulus=256)[:a_int.size] == (a_int ^ b_int)[:a_int.size]))


if __name__ == '__main__':
    main()
