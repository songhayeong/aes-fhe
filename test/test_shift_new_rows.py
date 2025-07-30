import numpy as np
import pytest

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.xor_service import XORConfig, EngineWrapper, XORService, ZetaEncoder, CoefficientCache
from aes_xor_fhe.new import AESFHERound, split_nibbles, decrypt_and_recombine


def pure_shiftrows(state: np.ndarray) -> np.ndarray:
    """
    4×4 AES ShiftRows in column-major flatten form.
    state: shape (16,), dtype uint8
    """
    assert state.shape == (16,)
    out = np.empty_like(state)
    # for each row r, column c
    for r in range(4):
        for c in range(4):
            idx = c * 4 + r
            new_c = (c + r) % 4
            new_idx = new_c * 4 + r
            out[new_idx] = state[idx]
    return out



@pytest.fixture(scope="module")
def aes_rounder():
    # 1) CKKS 엔진 세팅
    cfg = XORConfig(max_level=22, mode="parallel", thread_count=4, device_id=0)
    eng = EngineWrapper(cfg)
    # 2) XORService: 4→4 LUT 캐시 객체(니블 분할용) 삽입
    xor_svc = XORService(eng, coeff_cache=CoefficientCache(cfg.coeffs_path))  # 실제 캐시 클래스로 교체

    # 3) GF(2^8) ×2, ×3 호출 래퍼
    return AESFHERound(eng, xor_svc)


def test_shiftrows_only(aes_rounder):
    # 랜덤 16바이트 상태
    rng   = np.random.default_rng(123)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # 1) 평문 → 상하위 니블 분할
    hi, lo = split_nibbles(state)

    # 2) ζ₁₆ 인코딩 & 암호화
    z_hi = ZetaEncoder.to_zeta(hi, modulus=16)
    z_lo = ZetaEncoder.to_zeta(lo, modulus=16)
    ct_hi = aes_rounder.eng.encrypt(z_hi)
    ct_lo = aes_rounder.eng.encrypt(z_lo)

    # 3) FHE ShiftRows
    shr_hi, shr_lo = aes_rounder.shift_rows(ct_hi, ct_lo)

    # 4) 복호화 → 니블 재조합 → 바이트 복원
    out_bytes = decrypt_and_recombine(shr_hi, shr_lo, aes_rounder.eng, length=16)

    # 5) 순수 Python 기대 결과
    expected = aes_shiftrows_plain(state)

    assert np.array_equal(out_bytes, expected), (
        f"\nFHE ShiftRows out: {out_bytes}\nExpected:         {expected}"
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_shiftrows_simd(seed):
    # 1) 랜덤 16바이트 상태 생성
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # 2) hi/lo 니블 분할
    hi, lo = split_nibbles(state)

    # 3) ζ₁₆ 도메인 인코딩 & 암호화
    z_hi = ZetaEncoder.to_zeta(hi, modulus=16)
    z_lo = ZetaEncoder.to_zeta(lo, modulus=16)
    ctx = EngineContext(signature=2, use_bootstrap=True, mode="parallel", thread_count=8, device_id=0)
    enc_hi = ctx.engine.encrypt(z_hi, ctx.public_key)
    enc_lo = ctx.engine.encrypt(z_lo, ctx.public_key)

    # 4) FHE ShiftRows 수행
    out_hi, out_lo = AESFHERound.shift_rows(ctx, enc_hi, enc_lo)

    # 5) 복호화 & 디코딩
    dec_hi = ZetaEncoder.from_zeta(ctx.engine.decrypt(out_hi, ctx.secret_key), modulus=16)[:16]
    dec_lo = ZetaEncoder.from_zeta(ctx.engine.decrypt(out_lo, ctx.secret_key), modulus=16)[:16]

    # 6) 니블 재결합
    out_bytes = ((dec_hi.astype(np.uint8) << 4) | dec_lo.astype(np.uint8))

    # 7) 순수 파이썬 예상값
    expected = pure_shiftrows(state)

    assert np.array_equal(out_bytes, expected), (
        f"\nFHE ShiftRows out: {out_bytes}\nExpected:         {expected}"
    )