"""Make an offline, prerecorded viewer of the first evaluation map in every arm."""
import argparse
import copy
import gzip
import html as html_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.world import World


def export(folder, *, fixture=None, success_window=None, title='Godhood Trials · action-category comparison'):
    receipt = json.loads((folder/'audit.json').read_text())
    assert receipt['passed'], 'Export only audited evidence'
    protocol = json.loads((folder/'preregistration.json').read_text())
    chosen_seed = protocol['seeds'][0]
    episodes = {}; world = None
    with gzip.open(folder/'trace.jsonl.gz', 'rt', encoding='utf-8') as trace:
        for line in trace:
            row = json.loads(line)
            if row['type'] == 'start':
                meta = row['metadata']
                selected = (meta['stage'] == 'evaluation' and meta['case'] == 0 and meta['training_seed'] == chosen_seed
                            and (fixture is None or meta.get('fixture') == fixture))
                if not selected:continue
                world = World.from_dict(copy.deepcopy(row['world']))
                arm = meta['arm']; bounds = {}
                for rid, a in world.residents.items():
                    cells = [(x, y) for y in range(max(0, a.y-8), min(64, a.y+9))
                             for x in range(max(0, a.x-8), min(64, a.x+9)) if world.terrain[y][x] == 0]
                    bounds[rid] = [min(x for x, y in cells)-1, min(y for x, y in cells)-1,
                                   max(x for x, y in cells)+1, max(y for x, y in cells)+1]
                episode = dict(arm=arm, seed=chosen_seed, layout=meta['layout'], fixture=fixture, success_window=success_window,
                               terrain=world.terrain, bounds=bounds, frames=[])
                commands = {rid: {'verb': 'wait'} for rid in world.residents}
                def snapshot():
                    episode['frames'].append(dict(tick=world.tick, resources=copy.deepcopy(world.resources),
                        structures=copy.deepcopy(world.structures), bodies={rid: dict(x=a.x, y=a.y, food=a.food,
                            inventory=dict(a.inventory), unconscious=a.unconscious, action=commands[rid])
                            for rid, a in world.residents.items()}))
                snapshot()
            elif row['type'] == 'step' and world is not None:
                if world.tick % 4 == 0:commands = row['commands']
                result = world.step(row['commands'], scripted=False)
                assert json.dumps(result, sort_keys=True) == json.dumps(row['results'], sort_keys=True)
                if world.tick % 4 == 0:snapshot()
            elif row['type'] == 'end' and world is not None:
                assert json.dumps(world.to_dict(), sort_keys=True) == json.dumps(row['world'], sort_keys=True)
                episode['metrics'] = row['metrics']; episodes[arm] = episode; world = None
    assert set(episodes) == set(protocol['arms'])
    data = json.dumps(episodes, separators=(',', ':')).replace('</', '<\\/')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Godhood Trials · GT-01 recorded comparison</title><style>
:root{color-scheme:dark}body{background:#111821;color:#e5edf2;font:16px system-ui;margin:32px auto;max-width:980px;padding:0 20px}
h1{font-size:28px;margin-bottom:8px}p{color:#bcc9d6;line-height:1.5}button,select,input{font:inherit;margin:6px;padding:8px;border-radius:6px}
.controls{background:#1d2935;padding:12px;border-radius:10px;display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.rooms{display:flex;gap:24px;flex-wrap:wrap;margin-top:24px}.room{flex:1;min-width:270px}canvas{width:100%;max-width:400px;image-rendering:pixelated;border-radius:8px}
label{color:#c9d8e5}output{font-variant-numeric:tabular-nums}#seek{flex:1;min-width:150px}.result{color:#f4d789}small{color:#9ab0c2}
</style><h1>__HEADING__</h1>
<p>Prerecorded, audited evaluation from the first map of the first training seed. Switch between six conditions with identical starting worlds. These are fresh experimental residents. This page does not connect to or advance the live game.</p>
<div class="controls"><label>Condition <select id="arm"></select></label><button id="play">Play</button><label for="seek">Tick</label><input id="seek" type="range" min="0" max="128" value="0"><output id="tick"></output></div>
<div class="rooms"><div class="room"><h2>Resident r0</h2><canvas id="r0" width="360" height="360"></canvas><p id="info-r0"></p><p class="result" id="score-r0"></p></div>
<div class="room"><h2>Resident r1</h2><canvas id="r1" width="360" height="360"></canvas><p id="info-r1"></p><p class="result" id="score-r1"></p></div></div>
<p><small>Blue circle: resident. Pink: berries. Gold: dropped food. Green: planted crop. Dark gray: blocking terrain. Playback shows recorded physical state; the policies acted using their private senses. No tone meaning is assigned. One selected map illustrates behavior; use the full report for rates across all maps.</small></p>
<script>const DATA=__DATA__;
const select=document.querySelector('#arm'),seek=document.querySelector('#seek'),play=document.querySelector('#play');
for(const arm of Object.keys(DATA)){const o=document.createElement('option');o.value=arm;o.textContent=arm;select.append(o)}
let timer=null;function stop(){clearInterval(timer);timer=null;play.textContent='Play'}
function draw(){const e=DATA[select.value],frame=e.frames[+seek.value];seek.max=e.frames.length-1;document.querySelector('#tick').textContent=frame.tick+' / '+e.frames[e.frames.length-1].tick;
for(const rid of ['r0','r1']){const canvas=document.getElementById(rid),ctx=canvas.getContext('2d'),b=e.bounds[rid],size=360/(b[2]-b[0]+1),a=frame.bodies[rid];
ctx.clearRect(0,0,360,360);for(let y=b[1];y<=b[3];y++)for(let x=b[0];x<=b[2];x++){ctx.fillStyle=e.terrain[y][x]===0?'#405a43':'#25313c';ctx.fillRect((x-b[0])*size,(y-b[1])*size,size-1,size-1);
const r=frame.resources[x+','+y];if(r){ctx.fillStyle=r.kind==='berry'?'#dc7a9c':r.kind==='crop'?'#80bf65':'#dec178';ctx.beginPath();ctx.arc((x-b[0]+.5)*size,(y-b[1]+.5)*size,size*.2,0,Math.PI*2);ctx.fill()}}
ctx.fillStyle='#65c7f0';ctx.beginPath();ctx.arc((a.x-b[0]+.5)*size,(a.y-b[1]+.5)*size,size*.3,0,Math.PI*2);ctx.fill();
document.getElementById('info-'+rid).textContent='Fullness '+a.food.toFixed(2)+' · carrying '+a.inventory.food+' food, '+a.inventory.seed+' seeds · '+JSON.stringify(a.action);
const m=e.metrics[rid];let success=m.timely_acquisition,label='Full-life result';
if(e.success_window){label='First '+e.success_window+' ticks';success=m.first_meal_tick!==null&&m.first_meal_tick<e.success_window&&(m.first_zero_tick===null||m.first_meal_tick<m.first_zero_tick)&&(e.fixture==='carried'||(m.first_gather_food_tick!==null&&m.first_gather_food_tick<m.first_meal_tick));}
document.getElementById('score-'+rid).textContent=label+': '+(success?'feeding criterion met':'feeding criterion missed')+' · '+m.safe_eaten+' ordinary meals';}}
select.onchange=()=>{stop();seek.value=0;draw()};seek.oninput=()=>{stop();draw()};play.onclick=()=>{if(timer){stop();return}if(+seek.value===+seek.max)seek.value=0;play.textContent='Pause';timer=setInterval(()=>{seek.value=+seek.value+1;draw();if(+seek.value>=+seek.max)stop()},120)};draw();
</script></html>'''.replace('__DATA__', data).replace('__HEADING__', html_module.escape(title))
    path = folder/('replay-'+fixture+'.html' if fixture else 'replay.html'); path.write_text(html, encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local runs directory')
    print(export(folder))
