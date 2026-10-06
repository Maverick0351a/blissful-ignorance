# Hazards and fainting (v0.3)

Amber fruit gives 12 fullness and costs 20 health; ordinary food gives 25 fullness without harm. Gathering amber fruit is safe. Thorn contact costs 8 health with a 16-tick cooldown (four simulated seconds). Handling thorns also counts as contact. Health is floored at 1; mortality remains disabled.

At health <=20, residents and the player pass out. All actions are rejected while unconscious, and new visual memories stop forming. Pain fades 0.25 per tick. After 40 ticks without injury, unconscious recovery is 0.2 health/tick (awake recovery is 0.04). Wake at health >=30. Stationary unconscious bodies do not take repeated thorn damage; waking grants a 16-tick grace period to move away. These are explicit development rules, not medical realism. From the health floor, recovery takes roughly 46 simulated seconds. Pause also pauses recovery; fast-forward accelerates it.

Health, pain and unconscious state are personal body inputs. Visible plant categories are neutral cues, without toxicity or avoidance labels. Injury counts and causes in the journal are observer tools. Faint state and counters persist through saves. Existing worlds gain plants only on free grass, with existing resources, structures, residents and action RNG preserved.

The visible scripted baseline treats ordinary and amber food as possible food; it has no learned avoidance. The neural interface supports the new cues but remains untrained and disconnected. No observed behavior here establishes learning.

The next experiment should compare learning, frozen/untrained and scripted policies on fresh seeds and held-out maps, with both food types available. Measure harmful consumption per food-choice opportunity, thorn contacts per exposure, fainting, food acquisition and recovery together. Merely standing still, exhausting nearby hazards or changing maps is not evidence of learning. Health-floor saturation means future reward design must retain injury/pain signals as well as health change. Individual weights, recurrent states and learning histories must stay separate.

## Helping others

An awake resident with at least 10 energy can revive an unconscious cardinal neighbor using Help up / H. It costs 10 energy, restores consciousness and health to 30, leaves pain intact, and grants the same 16-tick thorn grace. It cannot heal an awake resident or operate remotely. Visible unconscious posture is a local sensory cue (schema 4); it reveals no identity or player flag. The scripted baseline helps adjacent unconscious neighbors automatically when it has enough energy. This is programmed assistance, not learned cooperation; distant rescue pathfinding is not implemented.
