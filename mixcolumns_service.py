from typing import Any
import numpy as np
from aes_xor_fhe.xor_service import ZetaEncoder, XORService, EngineWrapper
from aes_xor_fhe.gf_service import GFService


class AESFHETransformer:
    """
    CKKS SIMD 상에서 ShiftRows + MixColumns 및 역연산을 구현
    """

    def __init__(self,
                 engine_wrapper: EngineWrapper,
                 xor_svc: XORService,
                 gf_svc: GFService):
        self.eng = engine_wrapper
        self.xor_svc = xor_svc
        self.gf_svc = gf_svc

    def merged_shift_mix(self, state_bytes: np.ndarray) -> Any:
        eng = self.eng

        # 1) AES state -> zeta-domain encoding -> encrypt
        z = ZetaEncoder.to_zeta(state_bytes, modulus=256)
        ct = eng.encrypt(z)

        # 2) ShiftRows 관점에서 필요한 네 가지 회전만 미리 생성
        rts = {
            'A': ct,  # shift-0
            'A1': eng.rotate(ct, -1),  # Row1 shift : left shift 1
            'A6': eng.rotate(ct, -6),  # Row2 shift : left shift 2 -> overall -6 slot
            'A11': eng.rotate(ct, -11),  # Row3 shift : left shift 3 -> overall -11 slot
            # column-major flatten 기준으로 슬롯들을 각각 -1, -6, -11칸 회전한 암호문만 미리 만들어 둠.
        }

        # 3) "한 블록(Bi')"에 대응하는 MixColumns 계산 헬퍼
        # galois field에 대한 3,2,1 연산
        def apply_mix(spec: list[tuple[str, str]]) -> Any:
            """
            spec: [('A', 'mul1'), ('A1', 'mul2'), ('A6', 'mul3')] 같은 형태.
            mul1은 identity, mul2는 x2, mul3은 x3 in GF(2^8)
            """
            terms = []
            for rot_key, mul_fn in spec:
                c = rts[rot_key]
                if mul_fn == 'mul1':
                    terms.append(c)  # identity
                else:
                    terms.append(getattr(self.gf_svc, mul_fn)(c))  # x2, x3
            # XOR 합산
            acc = terms[0]
            for t in terms[1:]:
                acc = self.xor_svc.xor_cipher(acc, t)
            return acc

        # 4) AES MixColumns 4개 블록을 SIMD 방식으로 한꺼번에
        B0p = apply_mix([('A', 'mul2'),
                         ('A1', 'mul3'),
                         ('A6', 'mul1')])
        B1p = apply_mix([('A1', 'mul2'),
                         ('A6', 'mul3'),
                         ('A11', 'mul1')])
        B2p = apply_mix([('A6', 'mul2'),
                         ('A11', 'mul3'),
                         ('A1', 'mul1')])
        B3p = apply_mix([('A11', 'mul2'),
                         ('A1', 'mul3'),
                         ('A6', 'mul1')])

        # 5) 네 블록 최종 합체 (ShiftRows로 다시 rotate + XOR)
        out = B0p
        out = self.xor_svc.xor_cipher(out, eng.rotate(B1p, -1))
        out = self.xor_svc.xor_cipher(out, eng.rotate(B2p, -2))
        out = self.xor_svc.xor_cipher(out, eng.rotate(B3p, -3))

        return out

    def merged_inv_mixshift(self, ct_state: Any) -> Any:
        raise NotImplementedError(
            "InvMixColimns + InvShiftRows는 유사 방식으로 gf9, gf11, gf13, gf14 LUT 필요"
        )
