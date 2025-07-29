# test/test_engine_rot.py

import numpy as np
import pytest
from pathlib import Path

from aes_xor_fhe.xor_service import EngineWrapper, XORConfig


@pytest.fixture(scope="module")
def engine_wrap():
    # 기본 설정으로 Context/EngineWrapper 생성
    cfg = XORConfig(
        coeffs_path=Path("dummy"),
        nibble_lo_path=Path("dummy"),
        nibble_hi_path=Path("dummny")
    )
    return EngineWrapper(cfg)


def test_encrypt_decrypt_identity(engine_wrap):
    # 랜덤 실수 벡터 암호화→복호화 시 원본과 거의 동일해야 함
    sc = engine_wrap.engine.slot_count
    vec = np.linspace(0.0, 1.0, num=sc, dtype=np.float64)
    ct = engine_wrap.encrypt(vec)
    dec = engine_wrap.decrypt(ct)
    # CKKS 오차 범위 내
    assert np.allclose(dec, vec, atol=1e-6)


def test_rotate_shift(engine_wrap):
    # 슬롯 간 회전: plaintext 벡터를 직접 roll 한 것과 일치해야 함
    sc = engine_wrap.engine.slot_count
    base = np.arange(sc, dtype=np.float64)
    ct = engine_wrap.encrypt(base)
    # 오른쪽으로 5칸 회전
    ct_rot = engine_wrap.rotate(ct, 5)
    dec = engine_wrap.decrypt(ct_rot)
    assert np.allclose(dec, np.roll(base, 5), atol=1e-6)


def test_relinearize_noop(engine_wrap):
    # 신·구 암호문 차수 동일할 때 relinearize(no-op) 후에도 복호화 결과 동일
    sc = engine_wrap.engine.slot_count
    vec = np.random.RandomState(0).rand(sc)
    ct = engine_wrap.encrypt(vec)
    ct_relin = engine_wrap.relinearize(ct)  # 아무 연산 없이 호출해도
    dec = engine_wrap.decrypt(ct_relin)
    assert np.allclose(dec, vec, atol=1e-6)


def test_relinearize_after_mul(engine_wrap):
    # 곱셈 후 relinearize(차수 저감) → 복호화 결과가 제곱값과 일치해야 함
    sc = engine_wrap.engine.slot_count
    vec = np.random.RandomState(1).rand(sc)
    ct = engine_wrap.encrypt(vec)
    # ciphertext^2 (암호문끼리 곱) → relinearize
    ct_sq = engine_wrap.multiply(ct, ct, engine_wrap.relin_key)
    ct_sq_rl = engine_wrap.relinearize(ct_sq)
    dec = engine_wrap.decrypt(ct_sq_rl)
    assert np.allclose(dec, vec * vec, atol=1e-5)
