# test_aes_fhe_transformer.py
import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.xor_service import (
    EngineWrapper, XORConfig, CoefficientCache, XORService, ZetaEncoder, FullXORCache
)
from aes_xor_fhe.mixcolumns_service import AESFHETransformer


# --- 순수 파이썬 ShiftRows+MixColumns 구현 (AES 표준) ---
def xtime(a: int) -> int:
    """GF(2^8)에서 x2 (0x02) 곱셈"""
    return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else (a << 1)


def mix_single_column(col: np.ndarray) -> np.ndarray:
    """한 열(4바이트)에 대해 MixColumns 수행"""
    a0, a1, a2, a3 = col
    r0 = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3
    r1 = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3
    r2 = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3)
    r3 = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3)
    return np.array([r0, r1, r2, r3], dtype=np.uint8)


def aes_shift_mix_plain(state_bytes: np.ndarray) -> np.ndarray:
    """
    1) ShiftRows: 4x4 상태 행(row)마다 왼쪽으로 r칸 회전
    2) MixColumns: 각 열별로 GF(2^8) 다항식 곱+XOR
    state_bytes: 길이16, column-major 배열
    """
    # column-major → 4×4 행렬
    state = state_bytes.reshape((4, 4), order='F').copy()
    # ShiftRows
    for r in range(1, 4):
        state[r] = np.roll(state[r], -r)
    # MixColumns
    for c in range(4):
        state[:, c] = mix_single_column(state[:, c])
    # 다시 column-major 1D로
    return state.reshape(16, order='F')


@pytest.fixture(scope="module")
def transformer():
    # --- FHE 환경 초기화 ---
    ROOT = Path(__file__).resolve().parent.parent
    cfg = XORConfig(
        coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json",
        nibble_hi_path=ROOT / "generator/coeffs/nibble_hi_coeffs.json",
        nibble_lo_path=ROOT / "generator/coeffs/nibble_lo_coeffs.json",
        mul_coeffs_path=ROOT / "xor_256x256_coeffs.json"
    )
    eng_wrap = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    mul_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng_wrap, coeff_cache, hi_cache, lo_cache, mul_cache)
    return AESFHETransformer(eng_wrap, xor_svc)


def test_merged_shift_mix(transformer):
    rng = np.random.default_rng(1234)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # 1) FHE 연산
    ct_out = transformer.merged_shift_mix(state)

    # 2) 복호화·디코딩
    plain_z = transformer.eng.decrypt(ct_out)
    result = ZetaEncoder.from_zeta(plain_z, modulus=256)[:16].astype(np.uint8)

    # 3) 기대값 계산
    expected = aes_shift_mix_plain(state)

    assert np.array_equal(result, expected), (
        f"\nFHE result: {result}\n"
        f"Expected  : {expected}"
    )
