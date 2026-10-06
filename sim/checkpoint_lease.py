"""Exclusive ownership of a playable population checkpoint across processes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def checkpoint_lease(runs):
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / 'population-owner.json'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError as error:
        raise RuntimeError(f'Population already has an owner: {path}. '
                           'If a process crashed, verify its recorded PID has stopped before removing this file.') from error
    try:
        with handle:
            json.dump({'pid': os.getpid()}, handle)
        yield
    finally:
        path.unlink()
