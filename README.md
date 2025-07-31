# aes-fh
AES XOR FHE Library

This repository implements homomorphic AES operations using zeta-domain encoding and polynomial LUT approximations in CKKS.

Overview

We built a complete pipeline for evaluating AES rounds on encrypted data, including:
	1.	Zeta-domain Encoding
	•	Map integers (0–255) into roots of unity: ζ₂₅₆ = exp(-2πi·x/256).
	•	Maintain ciphertexts in the zeta domain for SIMD operations.
	2.	Nibble Splitting
	•	Split each byte into high and low 4-bit nibbles:

hi = byte >> 4
lo = byte & 0xF


	•	Encrypt ζ₁₆^hi and ζ₁₆^lo separately.

	3.	1D LUT Polynomial Approximation
	•	Generate IFFT-based coefficients for any 8→8 and 4→8 LUT by:

def compute_lut_coeffs(f, n):
    zeta_n = exp(-2πi/n)
    lut = [zeta_n**f(x) for x in range(n)]
    return np.fft.ifft(lut)


	•	Saved coefficients for GF multipliers (×2, ×3, ×9, ×11, ×13, ×14) and AES S-Box hi/lo.

	4.	GF(2⁸) Multipliers via 2-var Polynomials
	•	Implemented gf_mult_k(context, ct_hi, ct_lo) using sparse polynomial LUTs on (hi, lo) pairs.
	•	Avoid explicit bitwise extract by evaluating two small LUTs and combining.
	5.	ShiftRows + MixColumns Merger
	•	Precompute four rotated ciphertexts corresponding to AES ShiftRows shifts.
	•	Compute four SIMD blocks B0'..B3' by combining GF multipliers on rotated inputs.
	•	Final assemble: rotate back and XOR-merge to produce combined ShiftRows+MixColumns in one pass.
	6.	AddRoundKey Extension
	•	Homomorphic XOR-add round key by zeta-domain subtraction and addition.
	7.	Testing & Validation
	•	Unit tests comparing FHE outputs against pure-Python AES implementations.
	•	Debugged level counts and inserted bootstrapping where needed.

Directory Structure

aes_xor_fhe/
├── xor_service.py      # Zeta encode/decode, nibble split, XOR, AddRoundKey
├── gf_service.py       # GF multiplier wrappers (×2, ×3, …) via 2-var polynomials
├── mixcolumns_service.py  # Merged ShiftRows+MixColumns implementation
├── sbox_service.py     # AES S-Box via two 8→4 LUTs
├── generator/coeffs/    # Generated JSON coefficient files for LUTs
├── tests/               # pytest test suite for each component
└── README.md            # This document

Usage

from aes_xor_fhe import EngineWrapper, XORService, GFService, AESFHETransformer

# 1. Initialize FHE engine & services
generator = EngineWrapper(config)
xor_svc = XORService(generator)
gf_svc = GFService(generator, xor_svc)
transformer = AESFHETransformer(generator, xor_svc, gf_svc)

# 2. Encrypt and evaluate one AES round
state = np.random.randint(0, 256, size=16, dtype=np.uint8)
ct = transformer.merged_shift_mix(state)

# 3. Decrypt & decode
dec = generator.decrypt(ct)
out = XORService.from_zeta(dec, modulus=256)[:16]

Acknowledgments
	•	Based on polynomial LUT techniques for FHE-friendly AES (GHS-12 sketch).
	•	Uses DesiloFHE CKKS backend via EngineWrapper.

⸻

Happy Homomorphic AES!
