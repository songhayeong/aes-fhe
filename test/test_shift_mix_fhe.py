# test_shift_mix_fhe.py
from pathlib import Path

import numpy as np
import pytest

from aes_xor_fhe.xor_service import EngineWrapper, XORConfig, CoefficientCache, XORService, FullXORCache
from aes_xor_fhe.shift_mix_zeta import MixRow
from aes_xor_fhe.utils import zeta_decode


def aes_shift_mix_plain(state: np.ndarray) -> np.ndarray:
    """
    AES ShiftRows + MixColumns (GF(2^8)) 구현 (column-major).
    state: shape (4,4) uint8
    """
    # 1) ShiftRows
    sr = np.zeros_like(state)
    for r in range(4):
        sr[r] = np.roll(state[r], -r)

    # 2) MixColumns
    # multiplication in GF(2^8), using Rijndael's xtime
    def xtime(a):
        return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else (a << 1)

    out = np.zeros_like(sr)
    for c in range(4):
        a = sr[:, c]
        out[0, c] = xtime(a[0]) ^ (xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3]
        out[1, c] = a[0] ^ xtime(a[1]) ^ (xtime(a[2]) ^ a[2]) ^ a[3]
        out[2, c] = a[0] ^ a[1] ^ xtime(a[2]) ^ (xtime(a[3]) ^ a[3])
        out[3, c] = (xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ xtime(a[3])
    return out


def aes_inv_mixshift_plain(state: np.ndarray) -> np.ndarray:
    """
    AES InvMixColumns + InvShiftRows (GF(2^8)) 구현.
    """
    # 1) InvMixColumns matrix
    invM = np.array([
        [0x0E, 0x0B, 0x0D, 0x09],
        [0x09, 0x0E, 0x0B, 0x0D],
        [0x0D, 0x09, 0x0E, 0x0B],
        [0x0B, 0x0D, 0x09, 0x0E],
    ], dtype=np.uint8)

    # GF(2^8) mult
    def gmul(a, b):
        res = 0
        for i in range(8):
            if b & 1: res ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi: a ^= 0x1B
            b >>= 1
        return res

    # MixColumns⁻¹
    out_mc = np.zeros_like(state)
    for c in range(4):
        for r in range(4):
            s = 0
            for k in range(4):
                s ^= gmul(invM[r, k], state[k, c])
            out_mc[r, c] = s
    # InvShiftRows
    out = np.zeros_like(out_mc)
    for r in range(4):
        out[r] = np.roll(out_mc[r], r)
    return out


@pytest.fixture(scope="module")
def mixrow():
    ROOT = Path(__file__).resolve().parent.parent
    cfg = XORConfig(
        coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json",
        nibble_hi_path=ROOT / "generator/coeffs/nibble_hi_coeffs.json",
        nibble_lo_path=ROOT / "generator/coeffs/nibble_lo_coeffs.json",
        mul_coeffs_path=ROOT / "xor_256x256_coeffs.json"
    )
    eng_wrap = EngineWrapper(cfg)
    coeff = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    mul_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng_wrap, coeff, hi_cache, lo_cache, mul_cache)
    mr = MixRow(xor_svc, eng_wrap)
    return mr


def test_shift_mix_roundtrip(mixrow):
    # 랜덤 4×4 AES 상태
    rng = np.random.default_rng(2025)
    state = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)

    # 1) FHE 상에서 ShiftRows+MixColumns → InvMixColumns+InvShiftRows
    ct_forward = mixrow.merged_shift_mix_fhe(state_matrix=state)
    recovered = mixrow.merged_inv_mixshift_fhe_from_ct(ct_forward, mixrow)

    # 복구된 상태가 원본과 일치해야 한다
    assert np.array_equal(recovered, state)


def test_forward_matches_plain(fhe_env):
    engine, pk, sk, rk, rot_key, xor_svc = fhe_env

    rng = np.random.default_rng(2025)
    state = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)

    expected = aes_shift_mix_plain(state)

    # 2) FHE 상에서 같은 연산
    ct_forward = MixRow.merged_shift_mix_fhe(state)
    raw = engine.decrypt(ct_forward, sk)
    # ζ-domain → 정수 슬롯 복원
    slots = zeta_decode(raw, modulus=256)[:16].astype(np.uint8)
    out = slots.reshape((4, 4), order='C')

    assert np.array_equal(out, expected), f"\nFHE out: {out}\nexpected: {expected}"


def test_inverse_matches_plain(fhe_env):
    engine, pk, sk, rk, rot_key, xor_svc = fhe_env

    rng = np.random.default_rng(2025)
    state = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)

    expected = aes_inv_mixshift_plain(state)

    # 암호화 → FHE 상 역연산
    ct_forward = merged_shift_mix_fhe(state, engine, pk, rk, rot_key, xor_svc)
    ct_inv = merged_inv_mixshift_fhe_from_ct(ct_forward, engine, sk, rk, rot_key, xor_svc)

    # 복호화 & 디코딩
    raw = engine.decrypt(ct_inv, sk)
    slots = zeta_decode(raw, modulus=256)[:16].astype(np.uint8)
    out = slots.reshape((4, 4), order='C')

    assert np.array_equal(out, expected), f"\nFHE inv out: {out}\nexpected: {expected}"
