import numpy as np
from typing import Tuple
from aes_xor_fhe.xor_service import EngineWrapper, XORService, ZetaEncoder

class AESFHEShiftRows:
    """
    CKKS SIMD 상에서 column-major Flatten된 AES 상태에 대해
    ShiftRows (및 inverse) 연산을 수행하는 서비스
    """

    def __init__(self, engine_wrapper: EngineWrapper, xor_svc: XORService):
        self.eng = engine_wrapper
        self.xor_svc = xor_svc
        # rotation steps for each row in column-major (block size=16)
        # Row 0: no shift
        # Row 1: left shift by 1 -> rotate by -4
        # Row 2: left shift by 2 -> rotate by -8
        # Row 3: left shift by 3 -> rotate by -12
        self.row_rot = [0, -4, -8, -12]
        # Prepare masks plaintexts
        sc = self.eng.engine.slot_count
        # for each row, mask selects slots at indices row + 4*k for k=0..3
        masks = []
        for r in range(4):
            mask = np.zeros(16, dtype=float)
            mask[r::4] = 1.0
            # pad to full slot_count
            if mask.size < sc:
                mask = np.pad(mask, (0, sc - 16), constant_values=0.0)
            masks.append(self.eng.encode(mask))
        self.masks = masks

    def shift_rows(self, ct: object) -> object:
        """
        Flatten된 column-major AES state 1차원 ciphertext를 받아
        ShiftRows 연산을 진행 후 ciphertext 리턴
        """
        engine = self.eng
        # apply per-row mask + rotation, then recombine by homomorphic add
        out = None
        for r in range(4):
            # isolate row r
            ct_masked = engine.multiply(ct, self.masks[r])
            # rotate that row by the required step
            if self.row_rot[r] != 0:
                ct_rot = engine.rotate(ct_masked, self.row_rot[r])
            else:
                ct_rot = ct_masked
            # sum into output
            out = ct_rot if out is None else engine.add(out, ct_rot)
        return out

    def inverse_shift_rows(self, ct: object) -> object:
        """
        column-major flatten된 AES state ciphertext에 대해
        InvShiftRows 연산 수행 후 ciphertext 리턴
        """
        engine = self.eng
        # inverse shifts: row1 right by 1 -> rotate +4, row2 +8, row3 +12
        inv_rot = [0, +4, +8, +12]
        out = None
        for r in range(4):
            ct_masked = engine.multiply(ct, self.masks[r])
            if inv_rot[r] != 0:
                ct_rot = engine.rotate(ct_masked, inv_rot[r])
            else:
                ct_rot = ct_masked
            out = ct_rot if out is None else engine.add(out, ct_rot)
        return out
