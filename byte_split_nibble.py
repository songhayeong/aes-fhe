import numpy as np

def split_nibbles(flatten: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    :param flatten: np.ndarraym shape (N,), dtype uint8
        Input byte array
    :return:
    - upper : np.ndarray, shape (N, ), dtype uint8
        Upper four bits of every byte
    - lower : np.ndarray, shape (N, ), dtype uint8
        Lower four bits of every byte (values 0-15).
    """
    if flatten.dtype != np.uint8:
        flatten = flatten.astype(np.uint8, copy=False)

    upper = np.right_shift(flatten, 4, dtype=np.uint8) # flatten shift >> 4
    lower = np.bitwise_and(flatten, 0x0F, dtype=np.uint8) # flatten & 0x0F

    return upper, lower

