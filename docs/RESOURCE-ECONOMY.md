# Shared resources and farming

Implemented October 4, 2026. Laya remains a permanent supported architecture track; see [the architecture decision](ARCHITECTURE-DECISION.md).

**Historical milestone:** the rules and validation below record the original
October 4 farming implementation. The main valley now has eight independent
PPO residents and a distinct local NPU Laya; see [the population record](POPULATION-WATCH.md).
Current soil, moisture, irrigation and crop-yield rules are documented in
[resource ecology](LIVING-POPULATION.md#resource-rules). The scripted farming
routine described below is an older baseline, not the live controller.

- Wild fruit, wood and stone are finite. Automatic berry/amber refills are removed. Gathering the last unit frees the tile. Old exhausted bushes remain clearable through Remove.
- `make_seeds` consumes one ordinary food and produces two seeds. It needs one spare pack slot because total inventory grows by one. Amber fruit is not convertible.
- `plant` consumes one seed on empty grass at the actor's feet or one cardinal tile away. It cannot overwrite resources or structures. A seedling reserves that land until harvest; Remove does not erase growing crops.
- Each seed grows three food after 480 simulation ticks: two minutes at 1x, 24 seconds at 5x, six seconds at 20x. Pausing stops growth. One sacrificed fruit can ultimately yield six food if both seeds are planted and harvested.
- Crops are public resources. Any adjacent agent can take their fruit. Harvesting reduces the same authoritative stock for everyone. All agents observe before a tick; shuffled action order resolves competing claims without double awards.
- Residents see nearby seedlings and three broad growth stages. Exact maturity timestamps, planter identity and global food totals are not part of crop observations. The sidebar's totals are observer information only.
- Growth deadlines survive save/load. Existing structures, inventory and resource quantities load unchanged; the new finite-stock rule takes effect going forward.

## Controls

T / Make seeds, P / Plant at your feet, E / Gather. Seeds can also be given or dropped through the existing item selector. The help dialog explains the tradeoff. Seedlings have their own sprite and a lossy visual-memory cue.

## Behavior and limits

At this original milestone, the live baseline was scripted: fed, hydrated residents with spare food saved seeds, then planted on locally visible free grass. That demonstrated the economy without proving learned planning, cooperation, ownership, or competition strategy. Tiny policy experiments and frozen local Laya were then separate from the live population. No new model download or always-on Laya service was introduced by that update.

The neural scaffold now exposes crop presence/stage and six additional farming actions (conversion, planting here and four directions). Schema 5 has 32 visual channels, 79 other features, 51 actions and 148,116 parameters at width 128. Older neural checkpoints need explicit migration/retraining; world save schema remains compatible.

Food is renewable through deliberate planting; wood and stone are finite in this version. Water remains available from the river. Soil fertility, weather-dependent yields, spoilage and reforestation are future variables, not implemented claims. Mortality remains off, and hunger itself does not yet cause damage. Long-run scarcity and learned deferred gratification need new controlled experiments; prior foraging results used the previous resource rules.

## Verification

59 tests pass, including seed conversion/capacity, invalid planting without inventory loss, exact maturation timing, save replay, shared harvests, simultaneous last-fruit competition, visibility metadata, protected seedlings and absence of wild refills. Browser verification in a disposable world confirmed food-to-seed conversion, planting, updated inventory and crop totals; the 1280px viewport had no horizontal overflow. The live world is backed up before restart.
