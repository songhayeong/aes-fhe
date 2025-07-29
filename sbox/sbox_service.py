"""
    To-do
    - 엽력 바이트 (0~255)를 니블 단위로 분리
    - 각각에 대해 4-bit LUT(XORService pattern을 따름) 또는 미리 계산한 S-Box 다항식 계수 적용
    - 다시 합치는 과정 구현

    단위 테스트 작성 (test_sbox_service.py)
    - np.arrange(256) 같은 전체 바이트 값에 대해 S-Box 결과가 표준 AES S-Box와 일치하는지 검증

    간단한 벤치마크
    - 작은 벡터 (예 : 64byte) vs 큰 벡터 (32768byte)에서 정확도-속도 비교

    SubBytes (AES S-Box) homomorphic evaluation service.
    - Uses a dense 256-entry LUT-based polynomial in the zeta domain.
    - Computes coefficients via FFT once, the uses CKKS Engine.evaluate_polynomial

    -> 매번 계산하기엔 너무 비효율적이라 판단 따라서 generate_sbox_coeffs에서 미리 계산 후 json 형식으로 저장해 다시 불러오는 식으로 구현
    sbox는 항상 고정이기 때문
    AES SubBytes homomorphic evaluator loading precomputed 8 -> 4 LUT coefficients (Hi/Lo) from JSON
"""
import json
from functools import lru_cache
from typing import Any, List
from pathlib import Path

import numpy as np
from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.utils import zeta_encode

# AES standard S-Box
AES_SBOX = [
    # 0     1    2     3    4    5     6    7    8    9    A    B    C    D    E    F
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]


def load_json_coeffs(path: Path) -> np.ndarray:
    """
    Load LUT coefficients from JSON file (entries array of [i, real, imag]).
    Returns full-length numpy array of complex coefficients.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    n = data.get('n', None) or len(data['entries'])
    coeffs = np.zeros(n, dtype=np.complex128)
    for entry in data['entries']:
        i, real, imag = entry
        coeffs[int(i)] = real + 1j * imag
    return coeffs


class SBoxService:
    """
    AES SubBytes via two 8 -> 4 LUTs sharing a single power basis.
    """

    def __init__(
            self,
            ctx: EngineContext,
            hi_path: Path = Path(__file__).parent / 'coeffs' / 'sbox_hi_coeffs.json',
            lo_path: Path = Path(__file__).parent / 'coeffs' / 'sbox_lo_coeffs.json'
    ):
        self.ctx = ctx
        self.engine = ctx.engine
        self.rlk = ctx.relinearization_key
        # Load coefficients
        self.coeffs_hi = load_json_coeffs(hi_path)
        self.coeffs_lo = load_json_coeffs(lo_path)
        # Encode to plaintext
        sc = self.engine.slot_count
        self.pt_hi = [self.engine.encode(np.full(sc, c, dtype=np.complex128))
                      for c in self.coeffs_hi]
        self.pt_lo = [self.engine.encode(np.full(sc, c, dtype=np.complex128))
                      for c in self.coeffs_lo]

    def _build_power_basis(self, ct: Any) -> List[Any]:
        # builds [ct^1 , ... , ct^255]
        return self.engine.make_power_basis(ct, len(self.coeffs_hi) - 1, self.rlk)

    def sub_bytes(self, enc_byte: Any) -> Any:
        # Shared power basis
        powers = self._build_power_basis(enc_byte)
        # Hi nibble
        out_hi = self.engine.multiply(enc_byte, 0.0)
        for i, pt in enumerate(self.pt_hi):
            if abs(self.coeffs_hi[i]) < 1e-12:
                continue
            term = pt if i == 0 else self.engine.multiply(powers[i - 1], pt)
            out_hi = self.engine.add(out_hi, term)
        # Lo nibble
        out_lo = self.engine.multiply(enc_byte, 0.0)
        for i, pt in enumerate(self.pt_lo):
            if abs(self.coeffs_lo[i]) < 1e-12:
                continue
            term = pt if i == 0 else self.engine.multiply(powers[i - 1], pt)
            out_lo = self.engine.add(out_lo, term)
        # Combine hi/lo : multiply ciphertexts
        return self.engine.multiply(out_hi, out_lo, self.rlk)

    def sub_bytes_array(self, enc_arr: Any) -> Any:
        """
        SIMD SubBytes: apply AES S-Box to every slot in a ciphertext vector.
        Processes all slots in parallel by evaluating two 8→4 LUTs on the full ciphertext.
        """
        # Build power basis for full vector
        powers = self._build_power_basis(enc_arr)
        # Hi nibble
        out_hi = self.engine.multiply(enc_arr, 0.0)
        for i, pt in enumerate(self.pt_hi):
            if abs(self.coeffs_hi[i]) < 1e-12:
                continue
            term = pt if i == 0 else self.engine.multiply(powers[i - 1], pt)
            out_hi = self.engine.add(out_hi, term)
        # Lo nibble
        out_lo = self.engine.multiply(enc_arr, 0.0)
        for i, pt in enumerate(self.pt_lo):
            if abs(self.coeffs_lo[i]) < 1e-12:
                continue
            term = pt if i == 0 else self.engine.multiply(powers[i - 1], pt)
            out_lo = self.engine.add(out_lo, term)
        # Combine hi and lo parts
        return self.engine.multiply(out_hi, out_lo, self.rlk)

# @lru_cache(maxsize=1) 이제 json에서 직접 coeffs를 가져오기에 필요없음.
# def _compute_sbox_coeffs() -> List[complex]:
#     """
#     Compute dense LUT polynomial coefficients for AES S-Box via FFT:
#         lut[x] = zeta^SBOX[x] , zeta = exp(2j*pi/256)
#         coeffs = fft(lut)/256
#     """
#     n=256
#     # Use same base as zeta_encode : negative exponent
#     zeta = np.exp(-2j * np.pi / n)
#     # Build LUT values in zeta domain
#     lut = np.array([zeta ** AES_SBOX[x] for x in range(n)], dtype=np.complex128)
#     # Inverse DFT to get monomial coefficients
#     coeffs = np.fft.ifft(lut)
#     return coeffs.tolist()
#

# class SBoxService:
#     """
#     AES SubBytes homomorphic evaluator.
#     """
#     def __init__(self, ctx: EngineContext):
#         self.ctx = ctx
#         self.engine = ctx.engine
#         self.relin_key = ctx.relinearization_key
#         self.coeffs = _compute_sbox_coeffs()
#
#     def sub_bytes(self, enc_byte: Any) -> Any:
#         """
#         Apply AES S-Box LUT polynomial homomorphically to one ciphertext slot.
#         """
#         # Evaluate dense polynomial: Zeta^SBox[x] = f(Zeta^x)
#         return self.engine.evaluate_polynomial(
#             enc_byte,
#             self.coeffs,
#             self.relin_key
#         )
#
#     def sub_bytes_array(self, enc_err: Any) -> Any:
#         """
#         Apply S-Box to each slot in a SIMD-encrypted ciphertext array.
#         Assumes enc_arr packs up to slot_count bytes via zeta_encode.
#         """
#         # For full-array SIMD : requires one polynomial eval across all slots
#         return self.engine.evaluate_polynomial(
#             enc_err,
#             self.coeffs,
#             self.relin_key
#         )

# class SBoxService:
#     """
#     AES SubBytes homomorphic evaluator (sparse polynomial evaluation with complex coeff plaintexts)
#     위의 SBOX-Service에서는 복소수를 지원하지 않아 리팩토링.
#     - plaintext-encoded 복소계수 미리 계산.
#     - sub_bytes에서 power-basis를 만들고, 각 항을 평문 곱셈 후 암호문 덧셈으로 합산
#     - sub_bytes_array는 real-coeffs만 지원하는 SIMD용 예시
#     """
#     def __init__(self, ctx: EngineContext):
#         self.ctx = ctx
#         self.engine = ctx.engine
#         self.relin_key = ctx.relinearization_key
#         # Precompute plaintext-encoded coefficients
#         self.coeffs = _compute_sbox_coeffs() # list of complex coeffs length=256
#         self.pt_coeffs = [] # plaintext ciphertexts of coeffs
#         sc = self.engine.slot_count
#         for c in self.coeffs:
#             vec = np.full(sc, c, dtype=np.complex128)
#             self.pt_coeffs.append(self.engine.encode(vec))
#
#     def sub_bytes(self, enc_byte: Any) -> Any:
#         """
#         Apply AES S-Box LUT polynomial to a single-slot ciphertext.
#         """
#         # Build power basis : enc^1 ... enc^255
#         powers = self.engine.make_power_basis(enc_byte, len(self.coeffs)-1, self.relin_key)
#         # Constant term
#         result = self.engine.multiply(enc_byte, 0.0)
#         # Iterate through coefficients
#         for i, pt in enumerate(self.pt_coeffs):
#             if abs(self.coeffs[i]) < 1e-12:
#                 continue
#             if i == 0:
#                 # constant : add plaintext coeff
#                 term = self.engine.add(self.engine.multiply(enc_byte, 0.0), pt)
#             else:
#                 # power term : (enc^i) * coeff_pt
#                 power_ct = powers[i-1]
#                 term = self.engine.multiply(power_ct, pt)
#             result = self.engine.add(result, term)
#         return result
#
#     def sub_bytes_array(self, enc_arr: Any) -> Any:
#         """
#         SIMD evaluation on full ciphertext array: uses direct polynomial eval for real coeffs.
#         Note : may incur noise issues for large slot_count
#
#         # This will only work if coeffs are real; complex not supported by evaluate_polynomial
#         """
#         real_coeffs = [float(c.real) for c in self.coeffs]
