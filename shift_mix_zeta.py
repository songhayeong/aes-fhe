import numpy as np
from pathlib import Path

from aes_xor_fhe.wrapper import EngineWrapper
from aes_xor_fhe.utils import CoefficientCache, XORService, zeta_encode, zeta_decode


def merged_shift_mix_fhe(state_matrix, engine, pk, rk, rot_key, xor_svc: XORService):
    # 1. 4 * 4 상태 행렬 -> 16차원 벡터 (flatten)

    # 여기서 AES 상태 행렬을 펴서 길이 16짜리 1D 벡터로 바꿈!
    vec = np.array(state_matrix, dtype=np.float64).reshape(16, order='C')

    #실수 벡터를 DFT 슬롯 벡터로 바꿔줌
    zeta_vec = zeta_encode(vec)
    #그리고 CKKS 스킴에 따라, 이 복소수 슬롯 벡터 전체를 하나의 암호문에 담음!
    ct_state = engine.encrypt(zeta_vec, pk)

    X_fwd_vals = [
        [2, 3, 1, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2],
        [1, 2, 3, 1],
    ]
    ct_Xf = []
    for row in X_fwd_vals:
        row_vec = np.tile(row, 4).astype(np.float64)
        ct_Xf.append(engine.encrypt(zeta_encode(row_vec), pk))

    def compute_fwd_Bk(ct_s, ct_xk):
        Tb = engine.multiply(ct_s, ct_xk)
        Tb = engine.relinearize(Tb, rk)
 
        r1 = engine.rotate(Tb, rot_key, -1)
        r2 = engine.rotate(Tb, rot_key, -2)
        r3 = engine.rotate(Tb, rot_key, -3)
        c1 = xor_svc.xor_cipher(Tb, r1)
        c2 = xor_svc.xor_cipher(c1, r2)
        comp = xor_svc.xor_cipher(c2, r3)
        # 첫 슬롯만 추출
        mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
        ct_mask = engine.encrypt(zeta_encode(mask), pk)
        return engine.relinearize(engine.multiply(comp, ct_mask), rk)

    Bf_cts = [compute_fwd_Bk(ct_state, xk) for xk in ct_Xf]

    def collapse(ct_b):
        t2 = engine.rotate(ct_b, rot_key, -2)
        u1 = xor_svc.xor_cipher(ct_b, t2)
        t1 = engine.rotate(u1, rot_key, -1)
        u2 = xor_svc.xor_cipher(u1, t1)
        mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
        ct_mask = engine.encrypt(zeta_encode(mask), pk)
        return engine.relinearize(engine.multiply(u2, ct_mask), rk)

    Cf_cts = [collapse(b) for b in Bf_cts]

    shifts = [0, 5, 10, 15]
    out = None
    for k, ct in enumerate(Cf_cts):
        p = engine.rotate(ct, rot_key, -shifts[k])
        out = p if out is None else xor_svc.xor_cipher(out, p)
    return out


def merged_inv_mixshift_fhe_from_ct(ct_state, engine, sk, rk, rot_key, xor_svc: XORService):
    X_inv_vals = [
        [14, 11, 13, 9],
        [9, 14, 11, 13],
        [13, 9, 14, 11],
        [11, 13, 9, 14],
    ]
    ct_Xi = []

    pk_i = engine.create_public_key(sk)
    for row in X_inv_vals:
        row_vec = np.tile(row, 4).astype(np.float64)
        ct_Xi.append(engine.encrypt(zeta_encode(row_vec), pk_i))

    def compute_inv_Bk(ct_s, ct_xk):
        Tb = engine.multiply(ct_s, ct_xk)
        Tb = engine.relinearize(Tb, rk)
        r1 = engine.rotate(Tb, rot_key, -1)
        r2 = engine.rotate(Tb, rot_key, -2)
        r3 = engine.rotate(Tb, rot_key, -3)
        c1 = xor_svc.xor_cipher(Tb, r1)
        c2 = xor_svc.xor_cipher(c1, r2)
        comp = xor_svc.xor_cipher(c2, r3)
        mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
        ct_mask = engine.encrypt(zeta_encode(mask), pk_i)
        return engine.relinearize(engine.multiply(comp, ct_mask), rk)

    Bi_cts = [compute_inv_Bk(ct_state, xk) for xk in ct_Xi]

    def collapse_inv(ct_b):
        t2 = engine.rotate(ct_b, rot_key, -2)
        u1 = xor_svc.xor_cipher(ct_b, t2)
        t1 = engine.rotate(u1, rot_key, -1)
        u2 = xor_svc.xor_cipher(u1, t1)
        mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
        ct_mask = engine.encrypt(zeta_encode(mask), pk_i)
        return engine.relinearize(engine.multiply(u2, ct_mask), rk)

    Ci_cts = [collapse_inv(b) for b in Bi_cts]

    shifts = [0, 5, 10, 15]
    out = None
    for k, ct in enumerate(Ci_cts):
        p = engine.rotate(ct, rot_key, -shifts[k])
        out = p if out is None else xor_svc.xor_cipher(out, p)

    # 5) 복호화 및 디코딩
    raw = engine.decrypt(out, sk)
    # DFT 슬롯 벡터를 정수 슬롯 벡터로 바꾸어줌
    decoded = zeta_decode(raw)
    # 벡터 -> 4*4 상태 행렬로 재구성해줌
    vec = np.round(decoded).astype(np.int64)
    return vec.reshape((4, 4), order='C')


if __name__ == '__main__':
    cfg = EngineWrapper.XORConfig(
        coeffs_path=Path("xor_mono_coeffs.json"),
        nibble_hi_path=Path("nibble_hi_coeffs.json"),
        nibble_lo_path=Path("nibble_lo_coeffs.json")
    )
    eng_wrap = EngineWrapper(cfg)
    coeff_cache = CoefficientCache(cfg.coeffs_path)
    nib_hi_cache = CoefficientCache(cfg.nibble_hi_path)
    nib_lo_cache = CoefficientCache(cfg.nibble_lo_path)
    xor_svc = XORService(eng_wrap, coeff_cache, nib_hi_cache, nib_lo_cache)

    engine = eng_wrap.engine
    sk = eng_wrap.secret_key
    pk = eng_wrap.public_key
    rk = eng_wrap.relin_key
    rot_key = eng_wrap.conj_key

    # 테스트 평문
    A = [
        [0x01, 0x23, 0x45, 0x67],
        [0x89, 0xAB, 0xCD, 0xEF],
        [0x10, 0x32, 0x54, 0x76],
        [0x98, 0xBA, 0xDC, 0xFE],
    ]

    # 암호화 & 복호화 테스트
    ct = merged_shift_mix_fhe(A, engine, pk, rk, rot_key, xor_svc)
    rec = merged_inv_mixshift_fhe_from_ct(ct, engine, sk, rk, rot_key, xor_svc)
    print("Original:", np.array(A, dtype=np.int64))
    print("Recovered:", rec)
