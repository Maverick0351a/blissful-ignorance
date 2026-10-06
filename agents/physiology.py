"""Observed nutrition accounting, including decay clipped at zero fullness."""


def food_relief(previous, current, decay=.008):
    applied_decay = decay if current > 0 else min(decay, max(0., previous))
    relief = max(0., current-previous+applied_decay)
    return 0. if relief < 1e-7 else relief
