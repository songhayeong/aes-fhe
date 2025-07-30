# test_mixcolumns_fhe_transformer.py

import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.mixcolumns_service import AESFHETransformer
from aes_xor_fhe.xor_service import (
    EngineWrapper,
    XORConfig,
    CoefficientCache,
    XORService,
    ZetaEncoder, FullXORCache,
)
from aes_xor_fhe.gf_service import GFService


def aes_shift_mix_plain(state: np.ndarray) -> np.ndarray:
    """
    순수 파이썬으로 column-major 4×4 AES state에 대해
    ShiftRows → MixColumns 수행 후 다시 column-major로 flatten.
    """
    # 1) column-major reshape
    mat = state.reshape((4, 4), order='F').astype(np.uint8)

    # 2) ShiftRows: row r 를 왼쪽으로 r 만큼 순환 이동
    for r in range(4):
        mat[r] = np.roll(mat[r], -r)

    # 3) MixColumns (GF(2^8) 곱)
    def xtime(x: int) -> int:
        # finite-field ×2
        return (((x << 1) & 0xFF) ^ 0x1B) if (x & 0x80) else ((x << 1) & 0xFF)

    out = np.zeros_like(mat)
    for c in range(4):
        s0, s1, s2, s3 = mat[:, c]
        tmp = s0 ^ s1 ^ s2 ^ s3
        t0 = xtime(s0 ^ s1) ^ tmp ^ s0
        t1 = xtime(s1 ^ s2) ^ tmp ^ s1
        t2 = xtime(s2 ^ s3) ^ tmp ^ s2
        t3 = xtime(s3 ^ s0) ^ tmp ^ s3
        out[:, c] = [t0, t1, t2, t3]

    return out.flatten(order='F').astype(np.uint8)


@pytest.fixture(scope="module")
def transformer() -> AESFHETransformer:
    ROOT = Path(__file__).resolve().parent.parent
    cfg = XORConfig(
        coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json",
        nibble_hi_path=ROOT / "nibble_hi_coeffs.json",
        nibble_lo_path=ROOT / "nibble_lo_coeffs.json",
        mul_coeffs_path=ROOT / "xor_256x256_coeffs.json"
    )
    eng_wrap = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    mul_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng_wrap, coeff_cache, hi_cache, lo_cache, mul_cache)
    gf_svc = GFService(
        eng_wrap=eng_wrap,
        xor_svc=xor_svc,
        gf2_path=ROOT/"generator/coeffs/gf2_8to8_coeffs.json",
        gf3_path=ROOT/"generator/coeffs/gf3_8to8_coeffs.json"
    )

    return AESFHETransformer(engine_wrapper=eng_wrap, xor_svc=xor_svc, gf_svc=gf_svc)


def test_merged_shift_mix(transformer: AESFHETransformer):
    rng = np.random.default_rng(42)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # 1) homomorphic evaluate
    ct = transformer.merged_shift_mix(state)

    # 2) decrypt & zeta→정수 복원
    plain_z = transformer.eng.decrypt(ct)
    out = ZetaEncoder.from_zeta(plain_z, modulus=256)[:16].astype(np.uint8)

    # 3) 순수 파이썬 기대값
    expected = aes_shift_mix_plain(state)

    # 4) 비교
    assert np.array_equal(out, expected), (
        f"\nFHE out:      {out}\n"
        f"Expected plain:{expected}"
    )
