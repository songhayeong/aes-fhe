import numpy as np
from pathlib import Path

from aes_xor_fhe.xor_service import EngineWrapper, XORConfig, CoefficientCache, XORService
from aes_xor_fhe.utils import zeta_encode, zeta_decode


class MixRow:

    def __init__(self, xor_service: XORService, engine_wrapper: EngineWrapper):
        self.xor_svc = xor_service
        self.eng = engine_wrapper

    def merged_shift_mix_fhe(self, state_matrix):
        # 1. 4 * 4 상태 행렬 -> 16차원 벡터 (flatten)

        # 여기서 AES 상태 행렬을 펴서 길이 16짜리 1D 벡터로 바꿈!
        vec = np.array(state_matrix, dtype=np.float64).reshape(16, order='C')

        # 실수 벡터를 DFT 슬롯 벡터로 바꿔줌
        zeta_vec = zeta_encode(vec)
        # 그리고 CKKS 스킴에 따라, 이 복소수 슬롯 벡터 전체를 하나의 암호문에 담음!
        ct_state = self.eng.encrypt(zeta_vec)

        X_fwd_vals = [
            [2, 3, 1, 1],
            [1, 1, 2, 3],
            [3, 1, 1, 2],
            [1, 2, 3, 1],
        ]
        ct_Xf = []
        for row in X_fwd_vals:
            row_vec = np.tile(row, 4).astype(np.float64)
            ct_Xf.append(self.eng.encrypt(zeta_encode(row_vec)))

        def compute_fwd_Bk(ct_s, ct_xk):
            Tb = self.eng.multiply(ct_s, ct_xk)
            Tb = self.eng.relinearize(Tb)

            r1 = self.eng.rotate(Tb, -1)
            r2 = self.eng.rotate(Tb, -2)
            r3 = self.eng.rotate(Tb, -3)
            c1 = self.xor_svc.xor_cipher(Tb, r1)
            c2 = self.xor_svc.xor_cipher(c1, r2)
            comp = self.xor_svc.xor_cipher(c2, r3)
            # 첫 슬롯만 추출
            mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
            ct_mask = self.eng.encrypt(zeta_encode(mask))
            return self.eng.relinearize(self.eng.multiply(comp, ct_mask))

        Bf_cts = [compute_fwd_Bk(ct_state, xk) for xk in ct_Xf]

        def collapse(ct_b):
            t2 = self.eng.rotate(ct_b,-2)
            u1 = self.xor_svc.xor_cipher(ct_b, t2)
            t1 = self.eng.rotate(u1, -1)
            u2 = self.xor_svc.xor_cipher(u1, t1)
            mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
            ct_mask = self.eng.encrypt(zeta_encode(mask))
            return self.eng.relinearize(self.eng.multiply(u2, ct_mask))

        Cf_cts = [collapse(b) for b in Bf_cts]

        shifts = [0, 5, 10, 15]
        out = None
        for k, ct in enumerate(Cf_cts):
            p = self.eng.rotate(ct, -shifts[k])
            out = p if out is None else self.xor_svc.xor_cipher(out, p)
        return out

    def merged_inv_mixshift_fhe_from_ct(self, ct_state):
        X_inv_vals = [
            [14, 11, 13, 9],
            [9, 14, 11, 13],
            [13, 9, 14, 11],
            [11, 13, 9, 14],
        ]
        ct_Xi = []

        for row in X_inv_vals:
            row_vec = np.tile(row, 4).astype(np.float64)
            ct_Xi.append(self.eng.encrypt(zeta_encode(row_vec)))

        def compute_inv_Bk(ct_s, ct_xk):
            Tb = self.eng.multiply(ct_s, ct_xk)
            Tb = self.eng.relinearize(Tb)
            r1 = self.eng.rotate(Tb, -1)
            r2 = self.eng.rotate(Tb, -2)
            r3 = self.eng.rotate(Tb, -3)
            c1 = self.xor_svc.xor_cipher(Tb, r1)
            c2 = self.xor_svc.xor_cipher(c1, r2)
            comp = self.xor_svc.xor_cipher(c2, r3)
            mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
            ct_mask = self.eng.encrypt(zeta_encode(mask))
            return self.eng.relinearize(self.eng.multiply(comp, ct_mask))

        Bi_cts = [compute_inv_Bk(ct_state, xk) for xk in ct_Xi]

        def collapse_inv(ct_b):
            t2 = self.eng.rotate(ct_b, -2)
            u1 = self.xor_svc.xor_cipher(ct_b, t2)
            t1 = self.eng.rotate(u1, -1)
            u2 = self.xor_svc.xor_cipher(u1, t1)
            mask = np.array([1 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.float64)
            ct_mask = self.eng.encrypt(zeta_encode(mask))
            return self.eng.relinearize(self.eng.multiply(u2, ct_mask))

        Ci_cts = [collapse_inv(b) for b in Bi_cts]

        shifts = [0, 5, 10, 15]
        out = None
        for k, ct in enumerate(Ci_cts):
            p = self.eng.rotate(ct, -shifts[k])
            out = p if out is None else self.xor_svc.xor_cipher(out, p)

        # 5) 복호화 및 디코딩
        raw = self.eng.decrypt(out)
        # DFT 슬롯 벡터를 정수 슬롯 벡터로 바꾸어줌
        decoded = zeta_decode(raw)
        # 벡터 -> 4*4 상태 행렬로 재구성해줌
        vec = np.round(decoded).astype(np.int64)
        return vec.reshape((4, 4), order='C')
