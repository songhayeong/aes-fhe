import numpy as np

from aes_xor_fhe.gf_service import GFService
from aes_xor_fhe.xor_service import EngineWrapper, XORConfig, XORService, ZetaEncoder, CoefficientCache
from pathlib import Path
import time

np.random.seed(25073101)
state = np.random.randint(0, 255, 16, dtype=np.uint8)
print(state)

np.random.seed(25073102)
aes_key = np.random.randint(0,255,16,dtype=np.uint8)
print(aes_key)



def test_gf_lut_mul2_mul3():
    # 1) 난수 시드 & 입력 바이트 (0..255)
    rng = np.random.default_rng(123)
    state = rng.integers(0, 256, size=64, dtype=np.uint8)

    # 2) ζ₁₆ encoding
    z = ZetaEncoder.to_zeta(state % 16, modulus=16)

    ROOT = Path(__file__).resolve().parent.parent

    # 3) 엔진 초기화
    config = XORConfig(max_level=22, mode="parallel", thread_count=4, device_id=0,
                       coeffs_path=ROOT / "generator/coeffs/xor_mono_coeffs.json")
    eng_wrap = EngineWrapper(config)
    coeff_cache = CoefficientCache(config.coeffs_path)
    xor_svc = XORService(eng_wrap, coeff_cache)
    gf_svc = GFService(
        eng_wrap, xor_svc,
    )

    # 4) 암호화
    ct = eng_wrap.encrypt(z)

    # 5) GF×2, GF×3 연산
    start = time.time()
    hi2, lo2 = gf_svc.mul2(ct)
    hi3, lo3 = gf_svc.mul3(ct)
    print("GF LUT time:", time.time() - start)

    # 6) 복호화 & 디코딩
    dec_hi2 = ZetaEncoder.from_zeta(eng_wrap.decrypt(hi2), modulus=16).astype(int)
    dec_lo2 = ZetaEncoder.from_zeta(eng_wrap.decrypt(lo2), modulus=16).astype(int)
    dec_hi3 = ZetaEncoder.from_zeta(eng_wrap.decrypt(hi3), modulus=16).astype(int)
    dec_lo3 = ZetaEncoder.from_zeta(eng_wrap.decrypt(lo3), modulus=16).astype(int)




if __name__ == "__main__":
    test_gf_lut_mul2_mul3()
