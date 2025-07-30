# tests/test_shiftrows_service.py

import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.xor_service import EngineWrapper, XORConfig, CoefficientCache, XORService, ZetaEncoder, FullXORCache
from aes_xor_fhe.shiftrows_service import AESFHEShiftRows


def aes_shiftrows_plain(state: np.ndarray) -> np.ndarray:
    """
    순수 파이썬 column‐major ShiftRows:
    state를 (4×4) 매트릭스로 보고 각 행 r을 −r만큼 롤, 다시 펼침.
    """
    mat = state.reshape((4, 4), order="F")
    for r in range(4):
        mat[r] = np.roll(mat[r], -r)
    return mat.reshape(16, order="F")


@pytest.fixture(scope="module")
def fhe_services():
    # 기본 경로에 있는 JSON 파일을 그대로 사용
    cfg = XORConfig()
    eng = EngineWrapper(cfg)
    # XORService는 실제로 shift_rows에 쓰이진 않지만, 생성자 시 요구되므로 만들어 둡니다.
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    full_xor_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng, coeff_cache, hi_cache, lo_cache, full_xor_cache)
    return eng, xor_svc


def test_shift_and_inverse_shiftrows(fhe_services):
    engine, xor_svc = fhe_services열
    transformer = AESFHEShiftRows(engine, xor_svc)

    rng = np.random.default_rng(0)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # 1) ζ^(256) 도메인 인코딩 + 암호화
    z = ZetaEncoder.to_zeta(state, modulus=256)
    sc = engine.engine.slot_count
    if z.size < sc:
        z = np.pad(z, (0, sc - z.size), constant_values=1.0)
    ct = engine.encrypt(z)

    # 2) Homomorphic ShiftRows
    ct_sh = transformer.shift_rows(ct)

    # 3) 복호화 · 디코딩
    dec_sh = engine.decrypt(ct_sh)
    out_sh = ZetaEncoder.from_zeta(dec_sh, modulus=256)[:16].astype(np.uint8)

    # 4) 순수 파이썬 기대값 검사
    expected_sh = aes_shiftrows_plain(state)
    assert np.array_equal(out_sh, expected_sh), (
        f"\nFHE ShiftRows:  {out_sh}\nExpected ShiftRows: {expected_sh}"
    )

    # 5) Homomorphic InvShiftRows
    ct_inv = transformer.inverse_shift_rows(ct_sh)

    # 6) 복호화 · 디코딩 후 원상복구 확인
    dec_inv = engine.decrypt(ct_inv)
    out_inv = ZetaEncoder.from_zeta(dec_inv, modulus=256)[:16].astype(np.uint8)
    assert np.array_equal(out_inv, state), (
        f"\nAfter InvShiftRows: {out_inv}\nOriginal state:      {state}"
    )
