import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.xor_service import XORConfig, EngineWrapper, CoefficientCache, XORService, ZetaEncoder, FullXORCache
from aes_xor_fhe.mixcolumns_service import AESFHETransformer
from aes_xor_fhe.gf_service import GFService


def aes_shift_mix_plain(state: np.ndarray) -> np.ndarray:
    """
    Perform AES ShiftRows followed by MixColumns on a single 16-byte AES state
    given in **column-major** order.

    Parameters
    ----------
    state : np.ndarray, shape (16,), dtype uint8
        The input AES state flattened in column-major order:
        [s00, s10, s20, s30,  s01, s11, …, s33]

    Returns
    -------
    out : np.ndarray, shape (16,), dtype uint8
        The transformed state, still flattened column-major.
    """
    if state.shape != (16,) or state.dtype != np.uint8:
        raise ValueError("state must be a length-16 uint8 array")

    # 1) Reinterpret as 4×4 matrix in row-major by reading column-major:
    #    M[r,c] = state[c*4 + r]
    mat = state.reshape((4, 4), order='F').copy()

    # 2) ShiftRows: row r left-rotate by r
    for r in range(4):
        mat[r] = np.roll(mat[r], -r)

    # 3) MixColumns on each of the 4 columns
    def xtime(x):
        """GF(2^8) multiply by 2"""
        x = int(x)
        return ((x << 1) ^ 0x1B) & 0xFF if (x & 0x80) else (x << 1)

    def gf2(x):
        return xtime(x)

    def gf3(x):
        return xtime(x) ^ x

    out_mat = np.zeros_like(mat)
    for c in range(4):
        s0, s1, s2, s3 = mat[:, c]
        # according to AES spec
        out_mat[0, c] = gf2(s0) ^ gf3(s1) ^ s2 ^ s3
        out_mat[1, c] = s0 ^ gf2(s1) ^ gf3(s2) ^ s3
        out_mat[2, c] = s0 ^ s1 ^ gf2(s2) ^ gf3(s3)
        out_mat[3, c] = gf3(s0) ^ s1 ^ s2 ^ gf2(s3)

    # 4) Flatten back to column-major
    return out_mat.reshape(16, order='F').astype(np.uint8)


@pytest.fixture(scope="module")
def transformer():
    ROOT = Path(__file__).resolve().parent.parent
    cfg = XORConfig(
        coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json",
        nibble_hi_path=ROOT / "nibble_hi_coeffs.json",
        nibble_lo_path=ROOT / "nibble_lo_coeffs.json",
        mul_coeffs_path=ROOT / "xor_256x256_coeffs.json"
    )
    eng = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    hi_cache = CoefficientCache(cfg.nibble_hi_path)
    lo_cache = CoefficientCache(cfg.nibble_lo_path)
    mul_cache = FullXORCache(cfg.mul_path)
    xor_svc = XORService(eng, coeff_cache, hi_cache, lo_cache, mul_cache)

    # point at your 4→4 LUT JSONs for GF(2^8)×2 and ×3
    gf_svc = GFService(
        eng_wrap=eng,
        xor_svc=xor_svc,
        gf2_path=ROOT / "generator/coeffs/gf2_nibble_coeffs.json",
        gf3_path=ROOT / "generator/coeffs/gf3_nibble_coeffs.json",
    )

    return AESFHETransformer(engine_wrapper=eng, xor_svc=xor_svc, gf_svc=gf_svc)


def test_merged_shift_mix(transformer):
    # random AES state (column-major 4×4)
    rng = np.random.default_rng(42)
    state = rng.integers(0, 256, size=16, dtype=np.uint8)

    # homomorphic evaluate
    ct = transformer.merged_shift_mix(state)

    # decrypt & decode back to bytes
    plain_z = transformer.eng.decrypt(ct)
    out = ZetaEncoder.from_zeta(plain_z, modulus=256)[:16].astype(np.uint8)

    # compare to pure‐Python
    expected = aes_shift_mix_plain(state)
    assert np.array_equal(out, expected), (
        f"\nFHE out: {out}\nExpected: {expected}"
    )
