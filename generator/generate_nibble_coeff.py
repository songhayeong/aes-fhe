import json
from pathlib import Path
import numpy as np


def apply_conjugate_symmetry_1d(coeffs: np.ndarray) -> np.ndarray:
    """
    Enforce conjugate symmetry on a 1D coefficient vector:
    c[k] = conj(c[n-k]) and ensure middle term real if even.
    """
    n = coeffs.shape[0]
    res = coeffs.copy()
    for k in range(1, n // 2):
        avg = (res[k] + np.conj(res[n - k])) / 2
        res[k] = avg
        res[n - k] = np.conj(avg)
    if n % 2 == 0:
        res[n // 2] = res[n // 2].real + 0j
    return res


def compute_1d_lut_coeffs(output_func, n: int = 256, use_symmetry: bool = True) -> np.ndarray:
    """
    Compute 1D LUT polynomial coefficients for f(x) on 0..n-1 via ifft.
    Optionally apply conjugate symmetry.
    """
    zeta = np.exp(-2j * np.pi / n)
    lut = np.array([zeta ** output_func(x) for x in range(n)], dtype=np.complex128)
    coeffs = np.fft.ifft(lut)
    return apply_conjugate_symmetry_1d(coeffs) if use_symmetry else coeffs


def compute_2d_lut_coeffs(output_func, n: int = 16) -> np.ndarray:
    """
    Compute 2D LUT polynomial coefficients for f(i,j) on 0..n-1 via ifft2.
    Returns n×n array of complex coefficients.
    """
    zeta = np.exp(-2j * np.pi / n)
    lut2d = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            lut2d[i, j] = zeta ** output_func(i, j)
    # 2D IFFT includes 1/(n*n) scaling
    return np.fft.ifft2(lut2d)


def save_1d_coeffs(coeffs: np.ndarray, path: Path, tol: float = 1e-12):
    """
    Save unique 1D coefficients up to n/2 due to symmetry.
    JSON entries: [index, real, imag]
    """
    n = coeffs.shape[0]
    half = n // 2
    entries = []
    for k in range(half + 1):
        c = coeffs[k]
        if abs(c) > tol:
            entries.append([int(k), float(c.real), float(c.imag)])
    data = {"n": n, "tol": tol, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_2d_coeffs(coeffs: np.ndarray, path: Path, tol: float = 1e-12):
    """
    Save full 2D coefficients for 2D LUT.
    JSON entries: [i, j, real, imag]
    """
    n, m = coeffs.shape
    entries = []
    for i in range(n):
        for j in range(m):
            c = coeffs[i, j]
            if abs(c) > tol:
                entries.append([int(i), int(j), float(c.real), float(c.imag)])
    data = {"shape": [n, m], "tol": tol, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    base = Path(__file__).resolve().parent / "coeffs"
    # 1D nibble LUTs (256→16)
    hi = compute_1d_lut_coeffs(lambda x: x // 16)
    save_1d_coeffs(hi, base / "nibble_hi_coeffs.json")

    lo = compute_1d_lut_coeffs(lambda x: x % 16)
    save_1d_coeffs(lo, base / "nibble_lo_coeffs.json")

    # 2D XOR LUT (16×16)
    xor2d = compute_2d_lut_coeffs(lambda i, j: i ^ j)
    save_2d_coeffs(xor2d, base / "xor_mono_coeffs.json")

    print("Generated nibble_hi, nibble_lo (1D) and xor_mono (2D) coeffs")

if __name__ == "__main__":
    main()