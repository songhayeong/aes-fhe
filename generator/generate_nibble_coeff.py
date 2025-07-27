# generate_nibble_coeffs.py

import json
from pathlib import Path
import numpy as np


def compute_lut_coeffs(output_func, n=256):
    zeta = np.exp(-2j * np.pi / n)         # ζ256 domain
    lut  = np.array([zeta**output_func(x) for x in range(n)])
    return np.fft.ifft(lut)                # numpy.ifft includes 1/n scale


def save_coeffs(coeffs, path, tol=1e-12):
    entries = []
    for i, c in enumerate(coeffs):
        if abs(c) > tol:
            entries.append([i, float(c.real), float(c.imag)])
    data = {"n": len(coeffs), "tol": tol, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    base = Path(__file__).resolve().parent / "coeffs"
    # Hi nibble = floor(byte/16)
    hi = compute_lut_coeffs(lambda x: x // 16)
    # Lo nibble = byte % 16
    lo = compute_lut_coeffs(lambda x: x % 16)
    save_coeffs(hi, base/"nibble_hi_coeffs.json")
    save_coeffs(lo, base/"nibble_lo_coeffs.json")
    print("Generated nibble_hi_coeffs.json & nibble_lo_coeffs.json")


if __name__=="__main__":
    main()