# aes-fhe
AES FHE Library

# AES-XOR-FHE

Homomorphic evaluation of AES round functions (AddRoundKey, SubBytes, ShiftRows, MixColumns) over the CKKS scheme, using Zeta-domain encoding and lightweight LUT‐based GF(2⁸) multipliers.

---

## Table of Contents

- [Overview](#overview)  
- [Key Concepts](#key-concepts)  
- [Pipeline](#pipeline)  
- [Code Structure](#code-structure)  
- [Usage Example](#usage-example)  
- [Performance & Levels](#performance--levels)  
- [Future Work](#future-work)  

---

## Overview

This project implements **a full AES round** under CKKS homomorphic encryption by:

1. **Zeta-domain encoding** of bytes into complex “ζ” values  
2. **Nibble splitting** (high/low 4 bits) so that GF(2⁸) operations reduce to small‐domain LUTs  
3. **Polynomial evaluation** of 4→8 LUTs for GF×2, GF×3, etc. via precomputed coefficient JSONs  
4. **SIMD batching**: packing 2 K AES blocks per ciphertext  
5. **Merged ShiftRows+MixColumns** via only a handful of rotates, masks, and XORs  

---

## Key Concepts

- **Zeta Encoding**  
  Map each 0–15 nibble `x` to  
  ```python
  ζ₁₆^x = exp(-2j * π * x / 16)


⸻

Happy Homomorphic AES!
