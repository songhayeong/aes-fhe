import time
from pathlib import Path

import numpy as np

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.sbox.sbox_service import SBoxService
from xor_service import EngineWrapper, XORConfig, XORService, ZetaEncoder, CoefficientCache
from new import AESFHERound, split_nibbles

# 1) 난수 시드 고정 및 state/key 생성
np.random.seed(25073101)
state = np.random.randint(0, 256, 16, dtype=np.uint8)
print("Plaintext state:", state)

np.random.seed(25073102)
aes_key = np.random.randint(0, 256, 16, dtype=np.uint8)
print("AES key         :", aes_key)


def addroundkey():
    ROOT = Path(__file__).resolve().parent.parent

    # 2) CKKS 엔진 및 서비스 초기화
    config = XORConfig(max_level=30, mode="parallel", thread_count=4, device_id=0,
                       coeffs_path=ROOT /"aes_xor_fhe/generator/coeffs/xor_mono_coeffs.json")
    eng_wrap = EngineWrapper(config)
    coeff_cache = CoefficientCache(config.coeffs_path)
    xor_svc = XORService(eng_wrap, coeff_cache=coeff_cache)
    ark_svc = AESFHERound(eng_wrap, xor_svc)  # .full_round(state, key, recombine=True) 제공


    # 4) AddRoundKey 한 번 실행 & 시간 측정
    start = time.time()
    decoded = ark_svc.full_round(state, aes_key, recombine=True)
    elapsed = time.time() - start
    print(f"AddRoundKeyFHE time: {elapsed:.3f} seconds")

    # # 5) 복호화 & 디코딩
    # dec_z = eng_wrap.decrypt(ct_out)
    # decoded = ZetaEncoder.from_zeta(dec_z, modulus=256)[:16].astype(np.uint8)
    # print("Decrypted+XORed    :", decoded)

    # 6) 검증
    expected = (state ^ aes_key).astype(np.uint8)
    print("Expected XOR       :", expected)
    assert np.array_equal(decoded, expected), "AddRoundKey 결과 불일치!"
    print("✅ AddRoundKeyFHE 결과 일치")


def subbytes():
    # 2) nibble 분리 (0..255 → 상위/하위 4비트)
    st_hi, st_lo = split_nibbles(state)
    key_hi, key_lo = split_nibbles(aes_key)

    # 3) ζ-도메인 매핑 (modulus=16)
    #    SubBytes/LUT 구현 시 16원 루트 위에서 동작하기 때문입니다.
    z_hi = ZetaEncoder.to_zeta(st_hi, modulus=16)
    z_lo = ZetaEncoder.to_zeta(st_lo, modulus=16)
    kz_hi = ZetaEncoder.to_zeta(key_hi, modulus=16)
    kz_lo = ZetaEncoder.to_zeta(key_lo, modulus=16)

    # 4) CKKS 엔진 초기화
    ctx = EngineContext(
        signature=1,
        use_bootstrap=False,  # 부트스트랩 필요 없으면 False
        mode="parallel",
        thread_count=4,
        device_id=0
    )
    eng = ctx.engine

    # 5) 서비스 객체 생성
    xor_svc = XORService(eng_wrap=eng, coeff_cache=Path("path/to/nibble_coeffs.json"))
    sbox_svc = SBoxService(ctx)

    # 6) 암호화 & 시간 측정
    start = time.time()
    ct_hi = eng.encrypt(z_hi, ctx.public_key)
    ct_lo = eng.encrypt(z_lo, ctx.public_key)
    ckt_hi = eng.encrypt(kz_hi, ctx.public_key)
    ckt_lo = eng.encrypt(kz_lo, ctx.public_key)
    enc_time = time.time() - start

    print(f"Encryption took {enc_time * 1000:.2f} ms")
    print("Ciphertexts:\n", ct_hi, ct_lo, ckt_hi, ckt_lo)
