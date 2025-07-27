# tests/test_sbox_service.py

import pytest
import numpy as np
from pathlib import Path

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.sbox.sbox_service import SBoxService, load_json_coeffs, AES_SBOX
from aes_xor_fhe.utils import zeta_encode, zeta_decode

# Paths to the generated JSON coefficient files
HI_PATH = Path(__file__).resolve().parent.parent / "sbox" / "coeffs" / "sbox_hi_coeffs.json"
LO_PATH = Path(__file__).resolve().parent.parent / "sbox" / "coeffs" / "sbox_lo_coeffs.json"


@pytest.fixture(scope="module")
def ctx():
    # Use a small thread count for faster test runs
    return EngineContext(signature=2, max_level=22, mode="parallel", thread_count=8, device_id=0)


@pytest.fixture(scope="module")
def sbox_svc(ctx):
    return SBoxService(ctx, hi_path=HI_PATH, lo_path=LO_PATH)


def test_load_json_coeffs():
    hi = load_json_coeffs(HI_PATH)
    lo = load_json_coeffs(LO_PATH)
    assert hi.shape == lo.shape == (256,)
    assert np.count_nonzero(hi) > 0
    assert np.count_nonzero(lo) > 0


def test_coeffs_match_sbox():
    hi = load_json_coeffs(HI_PATH)
    lo = load_json_coeffs(LO_PATH)
    zeta4 = np.exp(-2j * np.pi / 16)
    for x in range(4):
        hi_val = int(zeta_decode(np.array([zeta4**hi[x]]), modulus=16)[0])
        lo_val = int(zeta_decode(np.array([zeta4**lo[x]]), modulus=16)[0])
        assert (hi_val << 4) | lo_val == AES_SBOX[x]


def test_sbox_mapping(sbox_svc):
    for x in range(256):
        za = zeta_encode([x], modulus=256)
        enc = sbox_svc.ctx.engine.encrypt(za, sbox_svc.ctx.public_key)
        out_ct = sbox_svc.sub_bytes(enc)
        dec = sbox_svc.ctx.engine.decrypt(out_ct, sbox_svc.ctx.secret_key)
        result = zeta_decode(dec, modulus=256)[0]
        assert result == AES_SBOX[x]


def test_sbox_array_simd(sbox_svc):
    ctx = sbox_svc.ctx
    slot_count = ctx.engine.slot_count
    base = np.arange(256, dtype=np.uint8)
    plaintexts = np.tile(base, slot_count // 256 + 1)[:slot_count]
    zeta_vec = zeta_encode(plaintexts, modulus=256)
    enc_vec = ctx.engine.encrypt(zeta_vec, ctx.public_key)
    out_vec_ct = sbox_svc.sub_bytes_array(enc_vec)
    dec = ctx.engine.decrypt(out_vec_ct, ctx.secret_key)
    results = zeta_decode(dec, modulus=256)
    expected = np.array([AES_SBOX[x] for x in plaintexts], dtype=np.uint8)
    assert np.array_equal(results, expected)