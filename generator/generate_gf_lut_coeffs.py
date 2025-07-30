# generate GF lut coeff
import json
import numpy as np
from pathlib import Path


# AES 다항식에 따라 GF(2^8)에서 곱하기 2와 곱하기 3을 구현

def gf_mul_2(byte: int) -> int:
    # 0x1B = x⁴ + x³ + x + 1
    b = byte << 1
    if byte & 0x80:
        b ^= 0x1B
    return b & 0xFF


def gf_mul_3(byte: int) -> int:
    return gf_mul_2(byte) ^ byte


# 2D LUT 계수 계산 함수 (출력은 16*16 배열)
def compute_2d_lut_coeffs(output_func, n=16) -> np.ndarray:
    zeta = np.exp(-2j * np.pi / n)
    # lut[i,j] = zeta ** output_func(16*i + j)
    lut = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            lut[i, j] = zeta ** output_func((i << 4) | j)
    # 2D IFFT (규모 1/(n²) 포함)
    return np.fft.ifft2(lut)


def save_2d_coeffs(coeffs: np.ndarray, path: Path, tol=1e-12):
    entries = []
    n, m = coeffs.shape
    for i in range(n):
        for j in range(m):
            c = coeffs[i, j]
            if abs(c) > tol:
                entries.append([i, j, float(c.real), float(c.imag)])
    data = {"shape": [n, m], "tol": tol, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(entries)} coeffs to {path}")


def main():
    base = Path(__file__).resolve().parent.parent / "aes_xor_fhe" / "generator" / "coeffs"

    # 1) gf_mul_2 에 대한 4-to-4 hi/lo LUT
    coeffs2 = compute_2d_lut_coeffs(gf_mul_2, n=16)
    save_2d_coeffs(coeffs2, base / "gf2_nibble_coeffs.json")

    # 2) gf_mul_3 에 대한 4-to-4 hi/lo LUT
    coeffs3 = compute_2d_lut_coeffs(gf_mul_3, n=16)
    save_2d_coeffs(coeffs3, base / "gf3_nibble_coeffs.json")


if __name__ == "__main__":
    main()
