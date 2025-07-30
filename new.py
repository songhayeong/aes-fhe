import numpy as np
from typing import Tuple, Any

from aes_xor_fhe.engine_context import EngineContext
from aes_xor_fhe.xor_service import EngineWrapper, XORService, ZetaEncoder, CoefficientCache


def _get_shift_rows_masks(ctx: EngineContext) -> dict[int, Any]:
    """
    row → Plaintext mask 캐시.
    키: 0,1,2,3 (row index)
    값: engine.encode된 boolean mask
    """
    if hasattr(ctx, "_sr_masks"):
        return ctx._sr_masks

    engine = ctx.engine
    sc = engine.slot_count
    # 블록 당 슬롯 수 (한 AES 블록이 차지하는 슬롯 개수)
    max_blocks = sc // 16

    masks: dict[int, Any] = {}
    for r in range(4):
        # row r에 해당하는 4개의 바이트(segment)를 1로, 나머지를 0으로
        # 전체 길이 = 16 * max_blocks
        before = 4 * r * max_blocks
        ones = 4 * max_blocks
        after = sc - before - ones
        arr = np.concatenate([
            np.zeros(before, dtype=float),
            np.ones(ones, dtype=float),
            np.zeros(after, dtype=float),
        ])
        masks[r] = engine.encode(arr)
    ctx._sr_masks = masks
    return masks

def split_nibbles(flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    평문 byte 배열을 상위/하위 니블로 분해
    :param flat: uint8 array shape (N,)
    :return: (upper, lower) each uint8 in 0..15
    """
    if flat.dtype != np.uint8:
        flat = flat.astype(np.uint8, copy=False)
    upper = np.right_shift(flat, 4, dtype=np.uint8)
    lower = np.bitwise_and(flat, 0x0F, dtype=np.uint8)
    return upper, lower


def decrypt_and_recombine(
        ct_hi: Any,
        ct_lo: Any,
        eng: EngineWrapper,
        length: int | None = None
) -> np.ndarray:
    """
    ct_hi/ct_lo 를 복호화하여 평문 바이트로 재조합
    """
    # 1) 복호화 → ζ₁₆ 벡터
    z_hi = eng.decrypt(ct_hi)
    z_lo = eng.decrypt(ct_lo)
    # 2) ζ₁₆ → 0..15 정수
    hi_vals = ZetaEncoder.from_zeta(z_hi, modulus=16)
    lo_vals = ZetaEncoder.from_zeta(z_lo, modulus=16)
    # 3) 필요 길이로 자르기
    if length is not None:
        hi_vals = hi_vals[:length]
        lo_vals = lo_vals[:length]
    # 4) 바이트 합성
    return ((hi_vals.astype(np.uint8) << 4) |
            lo_vals.astype(np.uint8))


class AESFHERound:
    """
    평문 state/key → FHE AddRoundKey → 평문 결과까지
    """

    def __init__(self, eng_wrap: EngineWrapper, xor_svc: XORService
                 # gf2, gf3에 그거 approx한거 넣어야함 gf_service !!
                 ):
        self.eng = eng_wrap
        self.xor = xor_svc
        self.masks = []
        sc = self.eng.engine.slot_count
        self.row_rot = [0, -4, -8, -12]

        for r in range(4):
            mask = np.zeros(16, dtype=float)
            mask[r::4] = 1.0
            if mask.size < sc:
                mask = np.pad(mask, (0, sc - 16), constant_values=0.0)
            self.masks.append(self.eng.encode(mask))

        # ShiftRows : column-major 4 x 4 state 에서 각 행마다 회전할 슬롯수
        # flat vector 상에서 한 칸 = 1

    def encrypt_nibbles(self, hi: np.ndarray, lo: np.ndarray) -> Tuple[Any, Any]:
        z_hi = ZetaEncoder.to_zeta(hi, modulus=16)
        z_lo = ZetaEncoder.to_zeta(lo, modulus=16)
        return self.eng.encrypt(z_hi), self.eng.encrypt(z_lo)

    def add_round_key(self, s_hi: Any, s_lo: Any, k_hi: Any, k_lo: Any) -> Tuple[Any, Any]:
        """동형 XOR (AddRoundKey)"""
        return (
            self.xor.xor_cipher(s_hi, k_hi),
            self.xor.xor_cipher(s_lo, k_lo)
        )





    def shift_rows(ctx: EngineContext, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        """
        FHE 상에서 4×4 AES-ShiftRows:
          row 0: shift 0
          row 1: shift 1
          row 2: shift 2
          row 3: shift 3
        """
        engine = ctx.engine
        sc = engine.slot_count
        max_blocks = sc // 16

        masks = _get_shift_rows_masks(ctx)

        out_hi, out_lo = None, None
        for r in range(4):
            # 1) row r 분리 (마스크)
            m_hi = engine.multiply(ct_hi, masks[r])
            m_lo = engine.multiply(ct_lo, masks[r])

            # 2) left shift by r positions → 전체 슬롯 단위로는 -r*max_blocks 회전
            if r != 0:
                steps = - r * max_blocks
                m_hi = engine.rotate(m_hi, steps)
                m_lo = engine.rotate(m_lo, steps)

            # 3) 누적 합산
            if out_hi is None:
                out_hi, out_lo = m_hi, m_lo
            else:
                out_hi = engine.add(out_hi, m_hi)
                out_lo = engine.add(out_lo, m_lo)

        return out_hi, out_lo

    def mix_columns(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        """니블 단위 MixColumns (GF(2^8)×2,×3 LUT 활용)"""
        # 네 개의 ShiftRows rotation 을 미리 생성
        base = {'A': (ct_hi, ct_lo)}
        flat = ct_hi  # 같은 로테이션으로 hi/lo 모두
        for name, rot in [('A1', -1), ('A6', -6), ('A11', -11)]:
            base[name] = (
                self.eng.rotate(ct_hi, rot),
                self.eng.rotate(ct_lo, rot)
            )

        def mix_spec(spec: list[Tuple[str, int]]) -> Tuple[Any, Any]:
            """spec = [('A',2),('A1',3),('A6',1)] 의 형태"""
            terms_hi = []
            terms_lo = []
            for blk, mul in spec:
                hi_ct, lo_ct = base[blk]
                if mul == 1:
                    terms_hi.append(hi_ct)
                    terms_lo.append(lo_ct)
                elif mul == 2:
                    h2, l2 = self.gf2(hi_ct, lo_ct)
                    terms_hi.append(h2);
                    terms_lo.append(l2)
                elif mul == 3:
                    h3, l3 = self.gf3(hi_ct, lo_ct)
                    terms_hi.append(h3);
                    terms_lo.append(l3)
            # XOR 합산
            acc_h = terms_hi[0]
            acc_l = terms_lo[0]
            for h, l in zip(terms_hi[1:], terms_lo[1:]):
                acc_h = self.xor.xor_cipher(acc_h, h)
                acc_l = self.xor.xor_cipher(acc_l, l)
            return acc_h, acc_l

    def full_round(
            self,
            state: np.ndarray,
            key: np.ndarray,
            recombine: bool = True
    ) -> Any:
        """
        :param state: uint8 array shape (N,)
        :param key:   uint8 array shape (N,)
        :param recombine: True 이면 복호화+재조합 결과(byte array) 반환
                          False 이면 (ct_hi, ct_lo) 반환
        """
        # 1) 니블 분해
        s_hi, s_lo = split_nibbles(state)
        k_hi, k_lo = split_nibbles(key)

        # 2) ζ₁₆ 인코딩
        z_s_hi = ZetaEncoder.to_zeta(s_hi, modulus=16)
        z_s_lo = ZetaEncoder.to_zeta(s_lo, modulus=16)
        z_k_hi = ZetaEncoder.to_zeta(k_hi, modulus=16)
        z_k_lo = ZetaEncoder.to_zeta(k_lo, modulus=16)

        # 3) 암호화 (SIMD slot_count 맞춰 패딩 자동 처리)
        ct_s_hi = self.eng.encrypt(z_s_hi)
        ct_s_lo = self.eng.encrypt(z_s_lo)
        ct_k_hi = self.eng.encrypt(z_k_hi)
        ct_k_lo = self.eng.encrypt(z_k_lo)

        # 4) 동형 XOR (AddRoundKey)
        ct_out_hi = self.xor.xor_cipher(ct_s_hi, ct_k_hi)
        ct_out_lo = self.xor.xor_cipher(ct_s_lo, ct_k_lo)

        if not recombine:
            return ct_out_hi, ct_out_lo

        # 5) 복호화 + 니블 재조합
        return decrypt_and_recombine(
            ct_out_hi,
            ct_out_lo,
            self.eng,
            length=state.shape[0]
        )


# 사용 예
if __name__ == "__main__":
    import time
    from aes_xor_fhe.xor_service import XORConfig

    # 1) 엔진 초기화
    cfg = XORConfig(
        max_level=22,
        mode="parallel",
        thread_count=4,
        device_id=0
    )
    eng_wrap = EngineWrapper(cfg)
    xor_svc = XORService(eng_wrap, coeff_cache=CoefficientCache(cfg.coeffs_path))  # coeff_cache 은 4→4 LUT 계수 캐시

    ark = AESFHERound(eng_wrap, xor_svc)

    # 2) 평문 state/key 준비
    rng = np.random.default_rng(1)
    size = 32768
    state = rng.integers(0, 256, size=size, dtype=np.uint8)
    key = rng.integers(0, 256, size=size, dtype=np.uint8)

    # 3) 시간 측정하며 한 라운드 실행
    start = time.time()
    result = ark.full_round(state, key, recombine=True)
    end = time.time()

    print("AddRoundKey (FHE) took", end - start, "seconds")
    # 4) 순수값 검증
    expected = state ^ key
    assert np.array_equal(result, expected), "AddRoundKey 결과 불일치"
    print("✔ OK")
