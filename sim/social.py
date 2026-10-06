"""Gentle local attention signals; no coercion, injuries or shared policy state."""
ATTENTION_TICKS = 16
TAP_COOLDOWN = 8
MAX_ATTENTION = 4
DIRECTIONS = {'north': (0, -1), 'east': (1, 0), 'south': (0, 1), 'west': (-1, 0)}
TONES = tuple(str(i) for i in range(10))
LEGACY_TONES = {'low': '0', 'mid': '1', 'high': '2'}
TONE_DURATION = 2  # Half a second at 1x, followed by at least one silent tick.
TONE_COOLDOWN = 3
AUDITORY_WINDOW = 80
AUDITORY_CAPACITY = 32


def sound_lifetime(kind):
    return TONE_DURATION if kind.startswith('tone_') else 8


def remember_tones(world):
    """Capture only tones actually heard here; stored identities never reach senses."""
    for rid, actor in world.residents.items():
        events = [e for e in world.auditory_memories.get(rid, [])
                  if 0 <= world.tick - e['heard_tick'] < AUDITORY_WINDOW]
        if not actor.unconscious:
            known = {tuple(e['key']) for e in events}
            for heard in world.hearing(actor, internal=True):
                if not heard['kind'].startswith('tone_') or tuple(heard['key']) in known:
                    continue
                events.append({'key': heard['key'], 'heard_tick': world.tick,
                               'tone': heard['kind'][5:], 'bearing': heard['bearing'],
                               'strength': heard['strength']})
                known.add(tuple(heard['key']))
        world.auditory_memories[rid] = events[-AUDITORY_CAPACITY:]


def auditory_memory(world, rid):
    return [{'tone': e['tone'], 'bearing': e['bearing'], 'strength': e['strength'],
             'age': world.tick - e['heard_tick']}
            for e in world.auditory_memories.get(rid, [])
            if 0 <= world.tick - e['heard_tick'] < AUDITORY_WINDOW][-AUDITORY_CAPACITY:]


def tone(world, rid, command):
    """Emit a meaningless acoustic symbol; meaning must come from experience."""
    actor = world.residents[rid]
    pitch = command.get('tone', '0')
    if isinstance(pitch, str):
        pitch = LEGACY_TONES.get(pitch, pitch)
    if not isinstance(pitch, str) or pitch not in TONES:
        return False, 'Choose one tone from 0 to 9.'
    if actor.unconscious:
        return False, 'Passed out. Advance time to recover.'
    if world.tick - actor.last_tone_tick < TONE_COOLDOWN:
        return False, 'Pause briefly before another tone.'
    world.emit_sound(actor, 'tone_' + pitch, 6)
    actor.last_tone_tick = world.tick
    actor.action = 'Sounding tone ' + pitch
    return True, 'Emitted tone ' + pitch + '.'


def attention(world, rid):
    """Return recipient-only copies without identity or absolute coordinates."""
    return [{'tick': e['tick'], 'bearing': e['bearing'], 'kind': 'tap'}
            for e in getattr(world, 'attention_events', {}).get(rid, [])
            if 0 <= world.tick - e['tick'] < ATTENTION_TICKS][-MAX_ATTENTION:]


def expire_attention(world):
    world.attention_events = {rid: events for rid in world.residents
                              if (events := attention(world, rid))}


def tap(world, rid, command):
    actor = world.residents[rid]
    if actor.unconscious:
        return False, 'Passed out. Advance time to recover.'
    if world.tick - getattr(actor, 'last_tap_tick', -1000) < TAP_COOLDOWN:
        return False, 'Give them a moment before tapping again.'

    def eligible(other):
        return (other.id != rid and not other.unconscious
                and abs(other.x-actor.x)+abs(other.y-actor.y) == 1
                and not world.opaque(actor.x, actor.y)
                and not world.opaque(other.x, other.y)
                and world.visible(actor, other.x, other.y))

    candidates = [a for a in world.residents.values() if eligible(a)]
    explicit = command.get('target')
    if explicit is not None:
        target = next((a for a in candidates if a.id == explicit), None)
    else:
        front = DIRECTIONS.get(actor.facing, (0, 1))
        candidates.sort(key=lambda a: ((a.x-actor.x, a.y-actor.y) != front, a.id))
        target = candidates[0] if candidates else None
    if target is None:
        return False, 'No conscious neighbor within clear arm reach.'
    bearing = next(d for d, delta in DIRECTIONS.items()
                   if delta == (actor.x-target.x, actor.y-target.y))
    if not hasattr(world, 'attention_events'):
        world.attention_events = {}
    events = attention(world, target.id)
    events.append({'tick': world.tick, 'bearing': bearing, 'kind': 'tap'})
    world.attention_events[target.id] = events[-MAX_ATTENTION:]
    actor.last_tap_tick = world.tick
    actor.action = 'Gently tapping a neighbor'
    target.facing = bearing
    target.action = 'Noticing a tap'
    world.record(f'{actor.name} gently tapped {target.name} for attention.', 'social', rid)
    return True, 'Gently tapped a neighbor for attention.'
