# test_recombine.py
import numpy as np
import pytest

from aes_xor_fhe.xor_service import ZetaEncoder
from aes_xor_fhe.new import decrypt_and_recombine


class DummyEngine:
    """decrypt(ct)가 그냥 ct를 반환하도록 하는 더미 엔진."""

    def decrypt(self, ct):
        return ct


def test_decrypt_and_recombine_full_length():
    # 1) 임의의 바이트 배열 준비
    rng = np.random.default_rng(0)
    state = rng.integers(0, 256, size=32768, dtype=np.uint8)

    # 2) 상/하위 니블 분리
    hi = state >> 4
    lo = state & 0x0F

    # 3) ζ₁₆ 도메인으로 인코딩
    z_hi = ZetaEncoder.to_zeta(hi, modulus=16)
    z_lo = ZetaEncoder.to_zeta(lo, modulus=16)

    # 4) decrypt_and_recombine 실행 (length=None → 전체 복원)
    dummy = DummyEngine()
    out = decrypt_and_recombine(z_hi, z_lo, dummy)

    # 5) 결과가 원본과 완전히 일치해야 한다
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, state)


def test_decrypt_and_recombine_partial_length():
    # 테스트용 짧은 배열
    state = np.array([0x00, 0x7F, 0xA5, 0xFF], dtype=np.uint8)
    hi = state >> 4
    lo = state & 0x0F
    z_hi = ZetaEncoder.to_zeta(hi, modulus=16)
    z_lo = ZetaEncoder.to_zeta(lo, modulus=16)

    dummy = DummyEngine()
    # length=2까지만 복원
    out2 = decrypt_and_recombine(z_hi, z_lo, dummy, length=2)
    assert np.array_equal(out2, state[:2])

    # length > 실제 길이면 에러보다는 가능한 만큼만 복원
    out5 = decrypt_and_recombine(z_hi, z_lo, dummy, length=10)
    assert np.array_equal(out5, state)  # 넘치는 부분은 잘릴 수 있음


if __name__ == "__main__":
    pytest.main([__file__])
