# test_merged_shift_mix_transformer.py
from pathlib import Path

import numpy as np
import pytest

from aes_xor_fhe.mixcolumns_service import AESFHETransformer
from aes_xor_fhe.xor_service import EngineWrapper, XORService, ZetaEncoder, CoefficientCache
from aes_xor_fhe.gf_service import GFService


# 순수‐파이썬 구현(import 경로는 프로젝트에 맞게 수정하세요)
def aes_shiftrows_plain(state: np.ndarray) -> np.ndarray:
    # state: 16바이트, column-major 순서
    st = state.reshape(4, 4)
    # ShiftRows
    for r in range(1, 4):
        st[r] = np.roll(st[r], -r)
    return st.reshape(16, order='F')


def mix_columns_numpy(arr: np.ndarray) -> np.ndarray:
    # arr: (4×4) column-major 블록
    def xtime(x):
        return (((x << 1) & 0xFF) ^ (((x >> 7) & 1) * 0x1B)).astype(np.uint8)

    s0, s1, s2, s3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    tmp = s0 ^ s1 ^ s2 ^ s3
    t0 = xtime(s0 ^ s1) ^ tmp ^ s0
    t1 = xtime(s1 ^ s2) ^ tmp ^ s1
    t2 = xtime(s2 ^ s3) ^ tmp ^ s2
    t3 = xtime(s3 ^ s0) ^ tmp ^ s3
    return np.stack([t0, t1, t2, t3], axis=1).astype(np.uint8)


def aes_shift_mix_plain(state: np.ndarray) -> np.ndarray:
    # apply ShiftRows → MixColumns to a single 4×4 block
    shifted = aes_shiftrows_plain(state)
    mat = shifted.reshape(4, 4).T  # convert to rows
    mixed = mix_columns_numpy(mat)
    return mixed.T.reshape(16, order='F')


@pytest.fixture(scope="module")
def transformer():
    # 여기에 실제 config 객체를 넣으세요
    from aes_xor_fhe.xor_service import XORConfig
    ROOT = Path(__file__).resolve().parent.parent

    config = XORConfig(

        max_level=22,
        mode="parallel",
        thread_count=8,
        device_id=0,
        coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json",
    )
    eng_wrap = EngineWrapper(config)
    coeff_cache = CoefficientCache(config.coeffs_path)
    xor_svc = XORService(eng_wrap, coeff_cache)
    gf_svc = GFService(eng_wrap, xor_svc)
    return AESFHETransformer(eng_wrap, xor_svc, gf_svc)


def test_single_random_state(transformer):
    rng = np.random.default_rng(123)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # homomorphic evaluation
    ct = transformer.merged_shift_mix(state)

    # decrypt & decode
    plain_z = transformer.eng.decrypt(ct)
    out = ZetaEncoder.from_zeta(plain_z, modulus=256)[:16].astype(np.uint8)

    # expected
    expected = aes_shift_mix_plain(state)

    assert np.array_equal(out, expected), (
        f"\nFHE out:      {out}\n"
        f"Expected:     {expected}"
    )


def test_multiple_random_states(transformer):
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = rng.integers(0, 256, size=16, dtype=np.uint8)
        ct = transformer.merged_shift_mix(state)
        out = ZetaEncoder.from_zeta(transformer.eng.decrypt(ct),
                                    modulus=256)[:16].astype(np.uint8)
        expected = aes_shift_mix_plain(state)
        np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    pytest.main([__file__])
