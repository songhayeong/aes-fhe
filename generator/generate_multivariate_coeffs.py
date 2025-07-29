# generator/generate_xor256_coeffs.py
import json
from pathlib import Path
import numpy as np


def compute_2d_lut_coeffs(output_func, n: int = 256) -> np.ndarray:
    zeta = np.exp(-2j * np.pi / n)
    lut2d = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            lut2d[i, j] = zeta ** output_func(i, j)
    return np.fft.ifft2(lut2d)  # 1/(n*n) 스케일 포함


def save_2d_coeffs(coeffs: np.ndarray, path: Path, tol: float = 1e-12):
    n, m = coeffs.shape
    entries = []
    for i in range(n):
        for j in range(m):
            c = coeffs[i, j]
            if abs(c) > tol:
                entries.append([i, j, float(c.real), float(c.imag)])
    data = {"shape": [n, m], "tol": tol, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "coeffs"
    xor256 = compute_2d_lut_coeffs(lambda a, b: a ^ b, n=256)
    save_2d_coeffs(xor256, base / "xor_256x256_coeffs.json")
    print("Generated full 8-to-8 XOR LUT (256×256) coefficients")
