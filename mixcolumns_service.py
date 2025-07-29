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
import numpy as np
from aes_xor_fhe.xor_service import ZetaEncoder, XORService, EngineWrapper


class AESFHETransformer:
    """
    CKKS SIMD 상에서 ShiftRows+MixColumns (및 역연산) 을 하나로 합친 구현.
    """

    def __init__(self, engine_wrapper: EngineWrapper, xor_svc: XORService):
        self.eng = engine_wrapper
        self.xor_svc = xor_svc

        # MixColumns 상수 마스크: 슬롯 0,4,8,12 에만 값이 있음
        mask1 = np.array([1 if i % 4 == 0 else 0 for i in range(16)], float)
        mask2 = mask1 * 2  # X = 2
        mask3 = mask1 * 3  # X+1 = 3

        # 미리 plaintext로 인코딩해 두기
        self.pt_C1 = self.eng.encode(ZetaEncoder.to_zeta(mask1, modulus=256))
        self.pt_CX = self.eng.encode(ZetaEncoder.to_zeta(mask2, modulus=256))
        self.pt_CX1 = self.eng.encode(ZetaEncoder.to_zeta(mask3, modulus=256))

    # 테스트 코드 필요함
    def merged_shift_mix(self, state_bytes: np.ndarray):
        # 1) AES state → ζ 인코딩 → 암호화
        z = ZetaEncoder.to_zeta(state_bytes, modulus=256)
        ct_A = self.eng.encrypt(z)

        # 2) ShiftRows 에 대응하는 3가지 회전
        ct_A1 = self.eng.rotate(ct_A, -1)
        ct_A6 = self.eng.rotate(ct_A, -6)
        ct_A11 = self.eng.rotate(ct_A, -11)

        # 3) 네 블록별 MixColumns
        # B0′ = A·C1 + A1·CX + A6·CX1
        B0p = self.eng.add(
            self.eng.multiply(ct_A, self.pt_C1, self.eng.relin_key),
            self.eng.add(
                self.eng.multiply(ct_A1, self.pt_CX, self.eng.relin_key),
                self.eng.multiply(ct_A6, self.pt_CX1, self.eng.relin_key),
            )
        )
        # B1′ = (A + A1)·C1 + A6·CX1 + A11·CX
        B1p = self.eng.add(
            self.eng.multiply(self.eng.add(ct_A, ct_A1), self.pt_C1, self.eng.relin_key),
            self.eng.add(
                self.eng.multiply(ct_A6, self.pt_CX1, self.eng.relin_key),
                self.eng.multiply(ct_A11, self.pt_CX, self.eng.relin_key),
            )
        )
        # B2′ = (A + A11)·C1 + A1·CX1 + A6·CX
        B2p = self.eng.add(
            self.eng.multiply(self.eng.add(ct_A, ct_A11), self.pt_C1, self.eng.relin_key),
            self.eng.add(
                self.eng.multiply(ct_A1, self.pt_CX1, self.eng.relin_key),
                self.eng.multiply(ct_A6, self.pt_CX, self.eng.relin_key),
            )
        )
        # B3′ = A·CX1 + A1·CX + (A6 + A11)·C1
        B3p = self.eng.add(
            self.eng.multiply(ct_A, self.pt_CX1, self.eng.relin_key),
            self.eng.add(
                self.eng.multiply(ct_A1, self.pt_CX, self.eng.relin_key),
                self.eng.multiply(self.eng.add(ct_A6, ct_A11), self.pt_C1, self.eng.relin_key),
            )
        )

        # 4) 네 블록 합쳐 최종 출력 (ShiftRows 후 MixColumns 결과)
        out = B0p
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B1p, -1))
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B2p, -2))
        out = self.xor_svc.xor_cipher(out, self.eng.rotate(B3p, -3))

        return out

    def merged_inv_mixshift(self, ct_state):
        pass