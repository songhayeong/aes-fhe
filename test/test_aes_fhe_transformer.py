# tests/test_aes_fhe_transformer.py
import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.xor_service import (
    XORConfig,
    EngineWrapper,
    CoefficientCache,
    XORService,
    ZetaEncoder,
    FullXORCache,
)
from aes_xor_fhe.shiftrow_mixcolumns import AESFHETransformer

# Paths to coefficient JSON files
ROOT = Path(__file__).resolve().parent.parent / "aes_xor_fhe"
XOR_COEFF_PATH = ROOT / "xor_mono_coeffs.json"
NIBBLE_HI_PATH = ROOT / "generator" / "coeffs" / "nibble_hi_coeffs.json"
NIBBLE_LO_PATH = ROOT / "generator" / "coeffs" / "nibble_lo_coeffs.json"
XOR_FULL_PATH = ROOT / "xor_256x256_coeffs.json"


@pytest.fixture(scope="module")
def aes_transformer():
    cfg = XORConfig()
    eng_wrap = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    full_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng_wrap, coeff_cache=coeff_cache, nibble_hi_path=hi_cache, nibble_lo_path=lo_cache,
                         full_xor_cache=full_cache)
    transformer = AESFHETransformer(xor_svc, eng_wrap)
    return transformer


def test_roundtrip_shiftmix(aes_transformer):
    # AES 테스트 벡터 (column-major 4×4)
    A = np.array([
        0x01, 0x23, 0x45, 0x67,
        0x89, 0xAB, 0xCD, 0xEF,
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xBA, 0xDC, 0xFE,
    ], dtype=np.uint8)

    # 1) ShiftRows+MixColumns
    ct_fwd = aes_transformer.merged_shift_mix(A)

    # 2) InvMixColumns+InvShiftRows
    ct_inv = aes_transformer.merged_inv_mixshift(ct_fwd)

    # 3) Decrypt & decode
    dec = aes_transformer.eng.decrypt(ct_inv)
    # Zeta 복호 → 정수, 원본 길이만큼 자르고 4×4 재배열
    vals = ZetaEncoder.from_zeta(dec, modulus=256)[:16]
    recovered = vals.reshape((4, 4), order="C")

    # 원본 A 도 4×4 형태로 비교
    expected = A.reshape((4, 4), order="C")
    assert np.array_equal(recovered, expected), f"round-trip failed: got\n{recovered}\nexpected\n{expected}"


def test_idempotent_forward_only(aes_transformer):
    # 작은 랜덤 상태로 forward만 수행해보고
    rng = np.random.default_rng(0)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)
    ct = aes_transformer.merged_shift_mix(state)
    # 단순히 decrypt 해보면, plaintext 단에서 ShiftRows+MixColumns를 직접 수행한 것과 일치해야 함
    dec = aes_transformer.eng.decrypt(ct)
    vals = ZetaEncoder.from_zeta(dec, modulus=256)[:16]

    # 순수 파이선 구현으로 검증
    def shift_rows(s):
        s = s.reshape((4, 4), order="C").copy()
        # AES ShiftRows
        s[1] = np.roll(s[1], -1)
        s[2] = np.roll(s[2], -2)
        s[3] = np.roll(s[3], -3)
        return s.flatten(order="C")

    def mix_cols(s):
        # AES MixColumns GF(2^8) 구현 (간단히 테이블 없이)
        def xtime(x): return ((x << 1) ^ 0x1B) & 0xFF if x & 0x80 else (x << 1)

        out = np.zeros(16, dtype=np.uint8)
        S = shift_rows(s)
        for c in range(4):
            col = S[c * 4:(c + 1) * 4]
            t = col[0] ^ col[1] ^ col[2] ^ col[3]
            u = col[0]
            out[c * 4 + 0] = col[0] ^ t ^ xtime(col[0] ^ col[1])
            out[c * 4 + 1] = col[1] ^ t ^ xtime(col[1] ^ col[2])
            out[c * 4 + 2] = col[2] ^ t ^ xtime(col[2] ^ col[3])
            out[c * 4 + 3] = col[3] ^ t ^ xtime(col[3] ^ u)
        return out

    # 직접 수행한 결과와 일치
    plain_expected = mix_cols(state)
    assert np.array_equal(vals, plain_expected)
