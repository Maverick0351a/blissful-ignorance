"""Preserve resident identity and commit decisions only as a complete group."""
from contextlib import contextmanager
import copy
import torch


def validate_roster(world, brains, metrics, neighbors_scripted, *, require_all=False):
    residents = set(world.residents) - {'player'}
    controlled = set(brains)
    if not controlled or not controlled <= residents:
        raise ValueError('Learner identities changed')
    if set(metrics) != controlled:
        raise ValueError('Learner metrics changed')
    if type(neighbors_scripted) is not bool:
        raise ValueError('Missing or invalid scripted-neighbor setting')
    # Isolated experiments intentionally keep resting bystanders. The playable
    # all-resident host must reject that shape unless migrating scripted neighbors.
    if require_all and not neighbors_scripted and controlled != residents:
        missing = ', '.join(sorted(residents - controlled))
        raise ValueError(f'Incomplete controlled population: missing brains for {missing}; no replacements created')
    if 'laya' in residents:
        brain = brains.get('laya')
        backend = brain.get('backend') if isinstance(brain, dict) else getattr(brain, 'backend', None)
        if backend != 'laya-npu':
            raise ValueError('Laya must retain her saved laya-npu controller; no replacement created')


def require_consistent(population):
    if getattr(population, 'recovery_required', None):
        raise RuntimeError(population.recovery_required)


def snapshot_state(state):
    # State contains detached checkpoint tensors, not live autograd graphs.
    # Clone each tensor once; avoid deepcopy's slower storage reconstruction.
    # The memo also preserves repeated references within pending/rollout history.
    memo = {}
    def clone_tensors(value):
        if id(value) in memo:
            return
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values(): clone_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value: clone_tensors(item)
    clone_tensors(state)
    return copy.deepcopy(state, memo)


@contextmanager
def decision_transaction(population):
    """Include optimizer tensors, private history and RNGs, not just counters.

    Observations are read-only and the world has not advanced at this point.
    Deep copies are essential: torch state_dict tensors alias live weights.
    """
    require_consistent(population)
    saved = {rid: snapshot_state(brain.state()) for rid, brain in population.brains.items()}
    try:
        yield
    except BaseException:
        failed = []
        for rid, brain in population.brains.items():
            try:
                brain.restore(saved[rid])
            except BaseException:
                failed.append(rid)
        if failed:
            population.recovery_required = (
                f'Decision rollback failed for {", ".join(failed)}. '
                'Saving and advancing are blocked; reload a complete checkpoint or restart.')
            raise RuntimeError(population.recovery_required)
        raise
