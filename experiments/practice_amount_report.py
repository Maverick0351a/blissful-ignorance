"""Local interactive summary and recorded replay, derived only after audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.category_replay import export


def report(folder):
    audit=json.loads((folder/'audit.json').read_text());assert audit['passed']
    result=json.loads((folder/'analysis.json').read_text())
    complete=json.loads((folder/'complete.json').read_text())
    training=json.loads((folder/'training.json').read_text())
    blocks={}
    for slot in range(6):
        cid=f'c{slot}';blocks[cid]=[]
        for index in range(4):
            items=[r['metrics'][rid] for r in training if index*16<=r['metadata']['case']<(index+1)*16
                for rid in ('r0','r1') if r['metadata']['candidates'][rid]==cid]
            assert len(items)==16
            blocks[cid].append(dict(lives_end=(index+1)*16,ordinary_meals=sum(m['safe_eaten'] for m in items),
                zero_food_ticks=sum(m['zero_food_ticks'] for m in items),
                mean_unique_tiles=sum(m['unique_tiles'] for m in items)/16))
    (folder/'training-blocks.json').write_text(json.dumps(blocks,indent=2),encoding='utf-8')
    payload=dict(result=result,blocks=blocks,seconds=complete['seconds'],audit_seconds=audit['seconds'])
    data=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Godhood Trials · Practice amount</title>
<style>:root{color-scheme:dark;--ink:#e9f0eb;--muted:#a8bbb4;--card:#172924;--line:#32473f}
*{box-sizing:border-box}body{margin:0;background:#0b1814;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1080px;margin:40px auto;padding:0 22px}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:8px 0 20px}
h2{font-size:22px;margin:0 0 14px}p{max-width:80ch;color:var(--muted)}a{color:#a1dbc4}
.eyebrow{color:#dfbd7a;letter-spacing:.12em;font-size:12px;text-transform:uppercase}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.card,section{padding:22px;border:1px solid var(--line);background:var(--card);border-radius:12px}.value{font-size:32px;font-weight:700}.sub{color:var(--muted);font-size:14px}
section{margin:20px 0}select{font:inherit;background:#0d1d17;color:var(--ink);border:1px solid #647d6e;padding:9px;border-radius:6px;margin:6px 16px 16px 8px}
.controls{display:flex;flex-wrap:wrap;align-items:center}.barrow{display:grid;grid-template-columns:100px 1fr 110px;align-items:center;gap:12px;margin:16px 0}.track{height:24px;background:#0c1c16;border-radius:6px;overflow:hidden}.bar{height:100%}.count{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;text-align:right;border-bottom:1px solid var(--line)}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}
.pass{color:#91dcad}.fail{color:#f1c080}ul{padding-left:20px}footer{font-size:13px;color:var(--muted);padding:20px 0 40px}
@media(max-width:650px){.cards{grid-template-columns:1fr}.barrow{grid-template-columns:80px 1fr 95px;font-size:14px}main{padding:0 14px}section{padding:16px}}
</style></head><body><main><div class="eyebrow">Godhood Trials · GT-01 · October 6, 2026</div>
<h1>Does more practice help?</h1><p>Six retained experimental learners. The same learning rule. Frozen evaluation after zero, sixteen and sixty-four additional practice lives.</p>
<div class="cards"><div class="card"><div class="sub">Prompt feeding after +64</div><div class="value" id="headline"></div><div class="sub">Fresh adjacent-food evaluation</div></div>
<div class="card"><div class="sub">Change versus +16</div><div class="value" id="difference"></div><div class="sub">Percentage points · same evaluation maps</div></div>
<div class="card"><div class="sub">Predeclared development screen</div><div class="value" id="verdict"></div><div class="sub">GT-01 remains open</div></div></div>
<section><h2>Feeding on fresh maps</h2><div class="controls"><label>Scope<select id="scope"><option value="all">All six learners</option></select></label>
<label>Task<select id="fixture"><option value="adjacent">Gather, then eat within 64 ticks</option><option value="carried">Eat carried fruit within 16 ticks</option></select></label></div>
<div id="bars"></div><p id="context"></p><div class="scroll"><table><thead><tr><th>Practice</th><th>Ordinary meals</th><th>Zero-food time</th><th>Unconscious time</th><th>Mean tiles visited</th></tr></thead><tbody id="details"></tbody></table></div>
<p>Each condition uses the same fresh maps and sampling seeds. No learning happens during evaluation. These small rooms place food within reach; the test does not measure broader navigation or sustained survival.</p></section>
<section><h2>Every learner</h2><div class="scroll"><table><thead><tr><th>Candidate</th><th>Starting</th><th>+16</th><th>+64</th><th>+64 minus +16</th></tr></thead><tbody id="candidates"></tbody></table></div>
<p>Scores are prompt gather-then-eat lives out of 16. Previously weak policy-gain responders c0, c3 and c5 were designated before this experiment; all six remain in the result.</p></section>
<section><h2>Screen and limits</h2><ul id="checks"></ul><p>Three historical seed groups, with two private brains each. More training changes both experience and update count. This is a descriptive practice-budget test; it does not rank architectures or prove a memory mechanism.</p>
<p>Farming and tones remain available. Their action counts alone do not establish productive farming or learned communication. No resident was culled, and the live population and Laya were not changed.</p></section>
<section><h2>Inspect the evidence</h2><p><a href="replay-adjacent.html">Play the first matched evaluation map</a> · <a href="analysis.json">All metrics</a> · <a href="training-blocks.json">Practice blocks</a> · <a href="audit.json">Independent audit</a> · <a href="preregistration.json">Protocol</a></p>
<p id="runtime"></p></section><footer>Offline report, built from audited recorded evidence. No live-game connection or external requests.</footer></main>
<script>const DATA=__DATA__;const arms=['starting','after-16','after-64'],labels=['Starting','+16 lives','+64 lives'],colors=['#8ca99b','#d0ac6b','#82cbaa'];
const $=id=>document.getElementById(id),rate=s=>s.prompt_successes/s.lives,pct=x=>(100*x).toFixed(1)+'%',signed=x=>(x>=0?'+':'')+x.toFixed(1)+' pp';
const overall=DATA.result.fixtures.adjacent.arms;
$('headline').textContent=overall['after-64'].prompt_successes+'/'+overall['after-64'].lives;
$('difference').textContent=signed(100*(rate(overall['after-64'])-rate(overall['after-16'])));
$('verdict').textContent=DATA.result.pilot_promising?'Passed':'Failed';
for(let i=0;i<6;i++){const o=document.createElement('option');o.value='c'+i;o.textContent='Candidate c'+i;$('scope').append(o)}
function render(){const f=$('fixture').value,scope=$('scope').value,s=DATA.result.fixtures[f],group=scope==='all'?s.arms:s.candidates[scope];
$('bars').innerHTML=arms.map((arm,i)=>{const x=group[arm];return '<div class="barrow"><span>'+labels[i]+'</span><div class="track"><div class="bar" style="background:'+colors[i]+';width:'+pct(rate(x))+'"></div></div><span class="count">'+x.prompt_successes+'/'+x.lives+' · '+pct(rate(x))+'</span></div>'}).join('');
$('context').textContent=scope==='all'?'96 resident lives per condition, grouped across three historical training seeds.':'16 resident lives per condition for '+scope+'.';
$('details').innerHTML=arms.map((arm,i)=>{const x=group[arm];return '<tr><td>'+labels[i]+'</td><td>'+x.ordinary_meals+'</td><td>'+pct(x.zero_food_ticks/x.resident_ticks)+'</td><td>'+pct(x.unconscious_ticks/x.resident_ticks)+'</td><td>'+x.mean_unique_tiles.toFixed(1)+'</td></tr>'}).join('');}
$('fixture').onchange=render;$('scope').onchange=render;render();
$('candidates').innerHTML=Object.entries(DATA.result.fixtures.adjacent.candidates).map(([cid,s])=>'<tr><td>'+cid+'</td>'+arms.map(a=>'<td>'+s[a].prompt_successes+'/16</td>').join('')+'<td>'+signed(100*(rate(s['after-64'])-rate(s['after-16'])))+'</td></tr>').join('');
const names={long_prompt_at_least_80_percent:'At least 80% prompt feeding after +64',long_exceeds_16_by_10_points:'At least 10 points above +16',long_exceeds_start_by_10_points:'At least 10 points above starting',positive_long_minus_16_in_at_least_two_seed_groups:'Improvement over +16 in at least two of three seed groups',carried_regression_no_more_than_5_points:'Carried-food decline no greater than five points',deprivation_increase_no_more_than_half_point:'Zero-food increase no greater than half a point'};
$('checks').innerHTML=Object.entries(DATA.result.checks).map(([k,v])=>'<li class="'+(v?'pass':'fail')+'">'+(v?'Pass: ':'Fail: ')+names[k]+'</li>').join('');
$('runtime').textContent='Run: '+DATA.seconds.toFixed(2)+' seconds. Separate replay audit: '+DATA.audit_seconds.toFixed(2)+' seconds. 181,248 world ticks; 49,152 training and 41,472 frozen decisions.';
</script></body></html>'''.replace('__DATA__',data)
    (folder/'report.html').write_text(page,encoding='utf-8')
    replay=export(folder,fixture='adjacent',success_window=64,title='Godhood Trials · retained learners and additional practice')
    content=replay.read_text(encoding='utf-8').replace('six conditions','three practice checkpoints').replace(
        'These are fresh experimental residents.',
        'These are retained experimental copies c0/r0 and c1/r1, evaluated with learning frozen.')
    replay.write_text(content,encoding='utf-8')
    return folder/'report.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local experiment directory')
    print(report(folder))
