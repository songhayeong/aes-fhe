from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from desilofhe import Engine


class EngineContext:
    """High-level container that owns an Engine and all related keys."""

    def __init__(self,
                 signature: int,
                 *,
                 max_level: int = 30,
                 mode: str = 'cpu',
                 use_bootstrap: bool = True,
                 use_multiparty: bool = False,
                 thread_count: int = 0,
                 device_id: int = 0,
                 fixed_rotation: bool = False,
                 delta_list: list[int] = None,
                 log_coeff_count: int = 0,
                 special_prime_count: int = 0) -> None:
        """Create an Engine and generate all default keys.

        init signature in desilo
        1. Engine(mode:str='cpu', use_bootstrap:bool=False, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        2. Engine(max_level: int, mode: str = ‘cpu’, *, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        3. Engine(log_coeff_count: int, special_prime_count: int, mode: str = ‘cpu’, *, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        """
        if signature == 1:
            self.engine = Engine(
                mode=mode,
                use_bootstrap=use_bootstrap,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id,
            )
        elif signature == 2:
            self.engine = Engine(
                max_level=max_level,
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id,
            )
        elif signature == 3:
            self.engine = Engine(
                log_coeff_count=log_coeff_count,
                special_prime_count=special_prime_count,
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id
            )
        else:
            raise ValueError(f"Unsupported signature: {signature}")

        self.fixed_rotation_key_list = []

        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relinearization_key = self.engine.create_relinearization_key(self.secret_key)
        self.conjugation_key = self.engine.create_conjugation_key(self.secret_key)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)

        if fixed_rotation and delta_list is not None:
            for delta in delta_list:
                self.fixed_rotation_key_list.append(self.engine.create_fixed_rotation_key(self.secret_key, delta))

        self.small_bootstrap_key = self.engine.create_small_bootstrap_key(self.secret_key)
        self.bootstrap_key = self.engine.create_bootstrap_key(self.secret_key)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FHEContext(engine=Engine(slot_count={self.engine.slot_count}), "
            f"keys=[sk, pk, rlk, cjk, rot])"
        )

    def encrypt(self, data):
        return self.engine.encrypt(data, self.public_key)

    def decrypt(self, ct):
        return self.engine.decrypt(ct, self.secret_key)