# GHS-12 기반 아이디어 스케치
"""
 1) state vector 준비
 - AES 상태 (4*4 byte)를 컬럼 바이트 순서 (column-major)로 16-길이 벡터 a에 담음.
   이를 Zeta-domain으로 Encoding -> Enc(zeta_encode(a))

2) 네 가지 회전 암호문 생성
- 논문에 따르면 네 개의 회전 automorphic (permutation) 만으로 ShiftRows를 대체가능.

# 원본 enc 상태
ct_A = enc
ct_A1 = engine.rotate(ct_A, -1) # 1칸 우측 회전 (InvShiftRows 관점에서는 +1)
ct_A6 = engine.rotate(ct_A, -6) # 6칸 우측 회전
ct_A11 = engine.rotate(ct_A, -11) # 11칸 우측 회전

이 넷이 각각 A1, A6, A11에 대응함.

3) MixColumns 계수 상수 준비
MixColumns를 수행할 때 네 개의 중간 블록마다 곱해야 할 상수 마스크 벡터 존재.
- C1 : 상수 "1"만 선택 (슬롯 0,4,8,12에만 1, 나머진 0)
- C_x : 상수 "X=2"
- C_x+1 : 상수 "3 = 2 xor 1"

이를 CKKS 상에서 plaintext 벡터로 인코딩

mask1 = np.array([1 if i%4==0 else 0 for i in range(16)], float)
maskX = mask1 * 2
maskX1 = mask1 * 3

pt_C1 = engine.encode(zeta_encode(mask1, modulus=256))
pt_CX   = engine.encode(zeta_encode(maskX, modulus=256))
pt_CX1  = engine.encode(zeta_encode(maskX1, modulus=256))

4) 네 블록별 MixColumns "한 번에 처리"
실제 MixColumns는 4*4 행렬 곱이지만, SIMD 슬롯 배열 덕분에 네 개의 블록(회전된 상태)를 동시에 처리할 수 있음.

# B0' = A * C1 + (A1) * CX + (A6) * CX1

# B1' = (A + A1) * C1 + A6*CX1 + A11*CX

# B2' = (A + A11) * C1 + A1 * CX1 + A6 * CX

# B3' = A * CX1 + A1 * CX + (A6 + A11) * C1

5) 네 블록을 합쳐 최종 출력
- 마지막으로 네 블록 B0' ... B3'를 다시 각각 0,1,2,3칸 회전하고 XOR으로 합치면 ShiftRows+MixColumns가 한 번에 끝남.

out = B0p
out = xor_cipher(out, engine.rotate(B1p, -1))
out = xor_cipher(out, engine.rotate(B2p, -2))
out = xor_cipher(out, engine.rotate(B3p, -3))

"""
from typing import Any

import numpy as np
from aes_xor_fhe.xor_service import ZetaEncoder, XORService, EngineWrapper
from aes_xor_fhe.gf_service import GFService


class AESFHETransformer:
    """
    CKKS SIMD 상에서 ShiftRows+MixColumns (및 역연산)을 한 번에 수행하는 구현.
    """

    def __init__(
            self,
            engine_wrapper: EngineWrapper,
            xor_svc: XORService,
            gf_svc: GFService,
    ):
        self.eng = engine_wrapper
        self.xor_svc = xor_svc
        self.gf_svc = gf_svc

    def merged_shift_mix(self, state_bytes: np.ndarray) -> Any:
        """
        Homomorphically evaluate ShiftRows+MixColumns on a single
        column-major 4×4 AES state (16 bytes).
        """
        # 1) ζ-domain 인코딩 → 암호화
        z = ZetaEncoder.to_zeta(state_bytes, modulus=256)
        ct = self.eng.encrypt(z)

        # 2) ShiftRows 대비 3가지 회전 (A1, A6, A11)
        ctA = ct
        ctA1 = self.eng.rotate(ctA, -1)
        ctA6 = self.eng.rotate(ctA, -6)
        ctA11 = self.eng.rotate(ctA, -11)

        # 3) MixColumns: GF(2^8) 곱은 반드시 LUT(mul1, mul2, mul3)로
        #   B0' = 1⊗A     ⊕ 2⊗A1   ⊕ 3⊗A6
        B0 = self.xor_svc.xor_cipher(
            self.gf_svc.mul1(ctA),
            self.gf_svc.mul2(ctA1)
        )
        B0 = self.xor_svc.xor_cipher(B0, self.gf_svc.mul3(ctA6))

        #   B1' = 1⊗(A⊕A1)  ⊕ 3⊗A6   ⊕ 2⊗A11
        t01 = self.xor_svc.xor_cipher(ctA, ctA1)
        B1 = self.xor_svc.xor_cipher(
            self.gf_svc.mul1(t01),
            self.gf_svc.mul3(ctA6)
        )
        B1 = self.xor_svc.xor_cipher(B1, self.gf_svc.mul2(ctA11))

        #   B2' = 1⊗(A⊕A11) ⊕ 3⊗A1   ⊕ 2⊗A6
        t011 = self.xor_svc.xor_cipher(ctA, ctA11)
        B2 = self.xor_svc.xor_cipher(
            self.gf_svc.mul1(t011),
            self.gf_svc.mul3(ctA1)
        )
        B2 = self.xor_svc.xor_cipher(B2, self.gf_svc.mul2(ctA6))

        #   B3' = 3⊗A     ⊕ 2⊗A1    ⊕ 1⊗(A6⊕A11)
        t611 = self.xor_svc.xor_cipher(ctA6, ctA11)
        B3 = self.xor_svc.xor_cipher(
            self.gf_svc.mul3(ctA),
            self.gf_svc.mul2(ctA1)
        )
        B3 = self.xor_svc.xor_cipher(B3, self.gf_svc.mul1(t611))

        # 4) 최종 ShiftRows 합산: B0 ⊕ rot⁻¹(B1) ⊕ rot⁻²(B2) ⊕ rot⁻³(B3)
        out = B0
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B1, -1))
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B2, -2))
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B3, -3))

        return out

    def merged_inv_mixshift(self, ct_state: Any) -> Any:
        raise NotImplementedError(
            "InvMixShiftRows 구현 시 mul9, mul11, mul13, mul14 LUT 필요"
        )
