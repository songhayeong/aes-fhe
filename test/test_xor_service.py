import numpy as np
import pytest
from pathlib import Path
from aes_xor_fhe.xor_service import XORService, EngineWrapper, XORConfig, CoefficientCache, ZetaEncoder, FullXORCache
from aes_xor_fhe.engine_context import EngineContext

# Paths to coefficient JSON files
ROOT = Path(__file__).resolve().parent.parent / "aes_xor_fhe"
XOR_COEFF_PATH = ROOT / "xor_mono_coeffs.json"
NIBBLE_HI_PATH = ROOT / "generator" / "coeffs" / "nibble_hi_coeffs.json"
NIBBLE_LO_PATH = ROOT / "generator" / "coeffs" / "nibble_lo_coeffs.json"
XOR_FULL_PATH = ROOT /  "xor_256x256_coeffs.json"


@pytest.fixture(scope="module")
def xor_svc():
    cfg = XORConfig()
    eng = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    full_cache = FullXORCache(cfg.mul_path)
    svc = XORService(eng, coeff_cache=coeff_cache, nibble_hi_path=hi_cache, nibble_lo_path=lo_cache, full_xor_cache=full_cache )
    return svc


def test_xor_simple(xor_svc):
    a = np.array([0, 1, 2, 3], dtype=np.uint8)
    b = np.array([3, 2, 1, 0], dtype=np.uint8)
    out = xor_svc.xor(a, b)
    assert np.array_equal(out, a ^ b)


def test_xor_random(xor_svc):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 16, size=32768, dtype=np.uint8)
    b = rng.integers(0, 16, size=32768, dtype=np.uint8)
    out = xor_svc.xor(a, b)
    assert np.array_equal(out, a ^ b)


def test_add_round_key_simd(xor_svc):
    # SIMD AddRoundKey for 8-bit values
    rng = np.random.default_rng(1)
    size = 32768
    state = rng.integers(0, 256, size=size, dtype=np.uint8)
    key = rng.integers(0, 256, size=size, dtype=np.uint8)

    # Encode state as full-slot zeta vector
    ze = ZetaEncoder.to_zeta(state, modulus=256)
    sc = xor_svc.eng.engine.slot_count
    if ze.size < sc:
        ze = np.pad(ze, (0, sc - ze.size), constant_values=1.0)
    enc_state = xor_svc.eng.encrypt(ze)

    # Perform AddRoundKey
    ark_ct = xor_svc.add_round_key(enc_state, key)

    # Decrypt and decode, then trim to original length
    dec = xor_svc.eng.decrypt(ark_ct)
    decoded = ZetaEncoder.from_zeta(dec, modulus=256)[:size]
    expected = state ^ key
    assert np.array_equal(decoded, expected)


def test_add_round_key_full(xor_svc):
    rng = np.random.default_rng(1)
    size = 32768
    state = rng.integers(0, 256, size=size, dtype=np.uint8)
    key = rng.integers(0, 256, size=size, dtype=np.uint8)

    # 1) state 를 ζ₍₂₅₆₎ 도메인으로 인코딩·암호화
    ze = ZetaEncoder.to_zeta(state, modulus=256)
    sc = xor_svc.eng.engine.slot_count
    ze = np.pad(ze, (0, sc - ze.size), constant_values=1.0)
    enc_state = xor_svc.eng.encrypt(ze)

    # 2) AddRoundKey = 풀 XOR 호출
    ark_ct = xor_svc.add_round_key_full(enc_state, key)

    # 3) 복호화 후 비교
    dec = xor_svc.eng.decrypt(ark_ct)
    out = ZetaEncoder.from_zeta(dec, modulus=256)[:size]
    assert np.array_equal(out, state ^ key)
    # 멈추는 현상 발생 ! sigkill 당한다...

def test_simple_test(xor_svc):
    rng = np.random.default_rng(1)
    size = 32768
    state = rng.integers(0, 256, size=size, dtype=np.uint8)
    key = rng.integers(0, 256, size=size, dtype=np.uint8)

    ze = ZetaEncoder.to_zeta(state, modulus=256)
    sc = xor_svc.eng.engine.slot_count
    if ze.size < sc:
        ze = np.pad(ze, (0, sc - ze.size), constant_values=1.0)
    enc_state = xor_svc.eng.encrypt(ze)

    ark = xor_svc.add_round_key()
