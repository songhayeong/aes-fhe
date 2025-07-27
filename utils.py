"""
 공통 유틸리티 함수 관리:
 - 바이트 블록 <-> AES 상태 배열 변환
 - Zeta (roots of unity) Encoding / Decoding
 - 패딩 및 블록 분할 등의 헬퍼
"""
from typing import Sequence
import numpy as np


def bytes_to_state(block: bytes) -> np.ndarray:
    """
    16 byte block을 AES state  4 * 4 matrix로 변환
    AES는 column-major mapping을 사용

    block[0] -> state[0, 0]
    block[1] -> state[1, 0]
    ...
    block[4] -> state[0, 1]
    """
    if len(block) != 16:
        raise ValueError("Block length must be 16 bytes")
    arr = np.frombuffer(block, dtype=np.uint8)
    # reshpae into (4, 4) column-major
    state = arr.reshape((4, 4), order='F')
    return state


def state_to_bytes(state: np.ndarray) -> bytes:
    """
    AES state matrix를 16byte block으로 변환
    column-major order(F-order)로 flatten
    """
    if state.shape != (4, 4):
        raise ValueError("State must be a 4x4 array")
    arr = state.reshape(16, order='F').astype(np.uint8)
    return arr.tobytes()


def zeta_encode(arr: Sequence[int], modulus: int = 16) -> np.ndarray:
    """
    정수 배열을 복소수 형태로 인코딩
    Zeta = exp(-2j*pi/modulus) (동형암호용 위상 인코딩)
    """
    arr_mod = np.asarray(arr, dtype=np.int64) % modulus
    zeta = np.exp(-2j*np.pi / modulus)
    return zeta ** arr_mod


def zeta_decode(z_arr: np.ndarray, modulus: int = 16) -> np.ndarray:
    """
    Zeta^k 형태의 복소수 배열을 정수 k로 디코딩
    위상을 측정하여 가장 가까운 정수로 반올림
    """
    angles = np.angle(z_arr)
    # 음수 위상이기에 부호 뒤집고 2pi를 나눠 정수 인덱스 계산
    idx = (-angles * modulus) / (2 * np.pi)
    idx_rounded = np.mod(np.rint(idx), modulus).astype(np.uint8)
    return idx_rounded


def chunk_bytes(data: bytes, block_size: int=16) -> list[bytes]:
    """
    데이터를 지정된 블록 크기로 분할하여 리스트를 반환
    마지막 블록은 패딩이 필요할 수 있음.
    """
    return [data[i:i+block_size] for i in range(0, len(data), block_size)]


def pkcs7_pad(block: bytes, block_size: int = 16) -> bytes:
    """
    PKCS#7 패딩을 적용하여 블록 길이를 맞춤.
    """
    pad_len = block_size - (len(block) % block_size)
    if pad_len == 0:
        pad_len = block_size
    return block + bytes([pad_len] * pad_len)


def pkcs7_unpad(data: bytes) -> bytes:
    """
    PKCS#7 패딩 제거
    """
    if not data:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > len(data):
        raise ValueError("Invalid padding")
    if data[-pad_len: ] != bytes([pad_len] * pad_len):
        raise ValueError("Invalid PKCS#7 padding bytes")
    return data[:-pad_len]