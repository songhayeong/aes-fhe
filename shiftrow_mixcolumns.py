import numpy as np
from pathlib import Path

from aes_xor_fhe.xor_service import ZetaEncoder, CoefficientCache, XORService, EngineWrapper


class AESFHETransformer:
    """
    Merge ShiftRows and MixColumns (and inverse) in SIMD over CKKS-encrypted state.
    """

    def __init__(self, xor_service: XORService, engine_wrapper: EngineWrapper):
        self.xor_svc = xor_service
        self.eng = engine_wrapper

    def merged_shift_mix(self, state_bytes: np.ndarray) -> object:
        """
        Apply ShiftRows followed by MixColumns homomorphically.
        - state_bytes : 1D_np.uint9 array of length 16 (column-major AES state)
        Returns a ciphertext encrypting the transformed state.
        """
        # 1) Zeta-domain encoding + encryption
        z_state = ZetaEncoder.to_zeta(state_bytes, modulus=256)
        enc_state = self.eng.encrypt(z_state)

        # 2) MixColumns 합성 매트릭스 (4x4)
        X = np.array([
            [2, 3, 1, 1],
            [1, 1, 2, 3],
            [1, 1, 2, 3],
            [3, 1, 1, 2]
        ], dtype=np.uint8)
        sc = self.eng.engine.slot_count
        ct_cols = []
        for row in X:
            vec = np.repeat(row, 4)
            z = ZetaEncoder.to_zeta(vec, modulus=256)
            if z.size < sc:
                z = np.pad(z, (0, sc - z.size), constant_values=1.0)
            ct_cols.append(self.eng.encrypt(z))

        # 3) 각 컬럼에 대해 MixColumns 연산
        Bf_cts = []
        for ct_x in ct_cols:
            # 곱셈
            Tb = self.eng.multiply(enc_state, ct_x, self.eng.relin_key)
            # 재선형화
            Tb = self.eng.relinearize(Tb)
            # XOR 합산: rotations + xor_cipher
            r1 = self.eng.rotate(Tb, -1)
            r2 = self.eng.rotate(Tb, -2)
            r3 = self.eng.rotate(Tb, -3)
            comp = self.eng.multiply(Tb, r1)
            comp = self.eng.multiply(comp, r2)
            comp = self.eng.multiply(comp, r3)
            # 첫 슬롯만 남기기 위한 마스크
            mask = np.array([1.0 if i % 4 == 0 else 0.0 for i in range(16)], dtype=np.uint8)
            z_mask = ZetaEncoder.to_zeta(mask, modulus=256)
            if z_mask.size < sc:
                z_mask = np.pad(z_mask, (0, sc - z_mask.size), constant_values=1.0)
            ct_mask = self.eng.encrypt(z_mask)
            Bf_cts.append(self.eng.multiply(comp, ct_mask, self.eng.relin_key))

        # 4) Collapse : Rotate-and-XOR to sum across rows
        Cf_cts = []
        for ct in Bf_cts:
            r2 = self.eng.rotate(ct, -2)
            u1 = self.xor_svc.xor_cipher(ct, r2)
            r1 = self.eng.rotate(u1, -1)
            u2 = self.xor_svc.xor_cipher(u1, r1)
            Ct = self.eng.multiply(u2, ct_mask, self.eng.relin_key)
            Cf_cts.append(self.eng.relinearize(Ct))

        # 5) 최종 ShiftRows: 슬롯 단위로 XOR 조합
        shifts = [0, 5, 10, 15]
        out = None
        for i, ct in enumerate(Cf_cts):
            p = self.eng.rotate(ct, -shifts[i])
            out = p if out is None else self.xor_svc.xor_cipher(out, p)
        return out

    def merged_inv_mixshift(self, enc_state: object) -> object:
        """
        Apply InvMixColumns followed by InvShiftRows on an encrypted state.
        - enc_state: ciphertext from merged_shift_mix
        Returns ciphertext of the inverse-transformed state.
        """
        # 1) Inverse MixColumns 매트릭스
        X_inv = np.array([
            [14, 11, 13, 9],
            [9, 14, 11, 13],
            [13, 9, 14, 11],
            [11, 13, 9, 14]
        ], dtype=np.uint8)
        sc = self.eng.engine.slot_count
        ct_cols = []
        for row in X_inv:
            vec = np.repeat(row, 4)
            z = ZetaEncoder.to_zeta(vec, modulus=256)
            if z.size < sc:
                z = np.pad(z, (0, sc - z.size), constant_values=1.0)
            ct_cols.append(self.eng.encrypt(z))

        Bi_cts = []
        for ct_x in ct_cols:
            Tb = self.eng.multiply(enc_state, ct_x, self.eng.relin_key)
            Tb = self.eng.relinearize(Tb)
            r1 = self.eng.rotate(Tb, -1)
            r2 = self.eng.rotate(Tb, -2)
            r3 = self.eng.rotate(Tb, -3)
            comp = self.xor_svc.xor_cipher(Tb, r1)
            comp = self.xor_svc.xor_cipher(comp, r2)
            comp = self.xor_svc.xor_cipher(comp, r3)
            Bi_cts.append(comp)

        # Collapse inverse
        Ci_cts = []
        for ct in Bi_cts:
            r2 = self.eng.rotate(ct, -2)
            u1 = self.xor_svc.xor_cipher(ct, r2)
            r1 = self.eng.rotate(u1, -1)
            u2 = self.xor_svc.xor_cipher(u1, r1)
            Ci_cts.append(self.eng.relinearize(u2))

        # InvShiftRows 합산
        shifts = [0, 5, 10, 15]
        out = None
        for i, ct in enumerate(Ci_cts):
            p = self.eng.rotate(ct, -shifts[i])
            out = p if out is None else self.xor_svc.xor_cipher(out, p)
        return out



