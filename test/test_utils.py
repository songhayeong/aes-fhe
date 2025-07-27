"""
    1. bytes_to_state <-> state_to_bytes 쌍이 상호 역함수 인지.
    2. zeta_encode <-> zeta_decode가 원래 정수로 정확히 복원되는지.
    3. pkcs7_pad / unpad 동작 검증
    4. chunk_bytes가 원하는 크기로 잘 자르는지

    를 검증하기 위한 모듈
"""

import pytest
import numpy as np
from aes_xor_fhe.utils import (
    bytes_to_state,
    state_to_bytes,
    zeta_encode,
    zeta_decode,
    chunk_bytes,
    pkcs7_pad,
    pkcs7_unpad,
)


def test_bytes_state_roundtrip():
    # 16-byte block of incremental values
    block = bytes(range(16))
    state = bytes_to_state(block)
    # state should be 4x4
    assert state.shape == (4,4)
    # roundtrip back to bytes
    out = state_to_bytes(state)
    assert out == block


def test_bytes_state_invalid_length():
    with pytest.raises(ValueError):
        bytes_to_state(b"short")
    with pytest.raises(ValueError):
        state_to_bytes(np.zeros((3,3), dtype=np.uint8))


def test_zeta_encode_decode():
    # test for all values 0..15
    arr = np.arange(16, dtype=np.int64)
    z = zeta_encode(arr, modulus=16)
    decoded = zeta_decode(z, modulus=16)
    assert np.array_equal(decoded, arr)

    # test random values
    rng = np.random.default_rng(123)
    arr2 = rng.integers(0, 16, size=100, dtype=np.int64)
    z2 = zeta_encode(arr2)
    dec2 = zeta_decode(z2)
    assert np.array_equal(dec2, arr2)


def test_chunk_bytes():
    data = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 26 bytes
    chunks = chunk_bytes(data, block_size=8)
    # expected 4 chunks: 8,8,8,2 bytes
    lengths = [len(c) for c in chunks]
    assert lengths == [8,8,8,2]
    # concatenation yields original
    assert b''.join(chunks) == data


def test_pkcs7_padding_unpadding():
    # exact multiple of block size
    block = b'HELLO_WORLD_1234'  # 16 bytes
    padded = pkcs7_pad(block, block_size=16)
    # pad length should be 16
    assert len(padded) == 32
    assert padded[-1] == 16
    unpadded = pkcs7_unpad(padded)
    assert unpadded == block

    # non-multiple length
    data = b'TEST'
    padded2 = pkcs7_pad(data, block_size=8)
    # pad length = 4 (to reach 8)
    assert len(padded2) == 8
    assert padded2[-1] == 4
    assert pkcs7_unpad(padded2) == data

    # invalid padding
    bad = b'data' + b'\x05\x05\x05\x05'
    with pytest.raises(ValueError):
        pkcs7_unpad(bad)

    bad2 = b'data' + b'\x00'
    with pytest.raises(ValueError):
        pkcs7_unpad(bad2)
