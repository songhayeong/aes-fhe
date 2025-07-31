# test_xor_service.py

import numpy as np
import pytest
from aes_xor_fhe.new import XORService, ZetaEncoder, CoefficientCache, AESFHERound


@pytest.fixture
def ark_svc():
    # 여기에 본인의 설정을 맞추어 주세요.
    # 예시:
    from aes_xor_fhe.xor_service import XORConfig, EngineWrapper
    config = XORConfig(
        max_level=22,
        mode="parallel",
        thread_count=4,
        device_id=0,
    )
    eng_wrap = EngineWrapper(config)
    xor = XORService(eng_wrap, coeff_cache=CoefficientCache(config.coeffs_path))
    return AESFHERound(eng_wrap, xor)


def test_add_round_key_simd(ark_svc):
    # SIMD AddRoundKey for 8-bit values
    rng = np.random.default_rng(1)
    size = 32768
    state = rng.integers(0, 256, size=size, dtype=np.uint8)
    key = rng.integers(0, 256, size=size, dtype=np.uint8)

    ct_full = ark_svc.full_round(state, key, recombine=True)

    # 3) Decrypt and decode, then trim to original length
    dec = ark_svc.eng.decrypt(ct_full)
    decoded = ZetaEncoder.from_zeta(dec, modulus=256)[:size].astype(np.uint8)

    # 4) Compare with numpy XOR
    expected = (state ^ key).astype(np.uint8)
    assert np.array_equal(decoded, expected), (
        f"\nDecoded : {decoded[:16]!r}…\nExpected: {expected[:16]!r}…"
    )
