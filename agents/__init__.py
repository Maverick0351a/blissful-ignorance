"""Optional neural scaffolding, initialized from scratch and currently untrained.

The stdlib simulation does not import this package. Importing this package alone
does not require PyTorch; requesting a model or controller does.
"""

__all__ = ["ResidentNetwork", "Controller", "encode_observation", "ACTIONS"]


def __getattr__(name):
    if name in __all__:
        from . import network
        return getattr(network, name)
    raise AttributeError(name)
