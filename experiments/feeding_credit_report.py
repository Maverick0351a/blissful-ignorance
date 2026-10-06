"""Create a standalone, offline view of audited feeding-update measurements."""
import argparse
import hashlib
import json
from pathlib import Path


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Godhood Trials · Feeding update diagnostic</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10191d;color:#e3ecee}
body{max-width:1120px;margin:auto;padding:28px}h1{font-size:clamp(25px,5vw,40px);margin:8px 0}
.eyebrow{letter-spacing:.12em;color:#93bac5;font-size:12px}.muted{color:#aebfc4;line-height:1.6}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}
.card,.panel{border:1px solid #34454c;border-radius:12px;background:#18252b;padding:20px}
.stat{font-size:30px;font-weight:650}.label{font-size:13px;color:#b4c5cb}.panel{margin:20px 0}
select{background:#0d171b;color:#fff;border:1px solid #77979f;border-radius:6px;padding:8px;font:inherit}
svg{display:block;width:100%;height:auto}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}
.legend span:before{content:'●';color:var(--ink);margin-right:7px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:13px 10px;border-bottom:1px solid #34454c;text-align:right}
th:first-child,td:first-child{text-align:left}th{color:#adc9d3}a{color:#89cfeb}.note{font-size:13px}
@media(max-width:680px){body{padding:16px}.cards{grid-template-columns:1fr}.card{padding:14px}.panel{padding:12px}}
</style>
<div class="eyebrow">GODHOOD TRIALS / GT-01 / OCTOBER 6, 2026</div>
<h1>What happens after a meal?</h1>
<p class="muted">A replay of six experimental category-PPO brains' existing training. Every native action,
update and final learning state reproduced exactly. This is a diagnostic of recorded learning;
it is not a new survival trial. The eight live PPO residents and Laya were preserved.</p>
<div class="cards"><div class="card"><div class="stat" id="credit"></div><div class="label">meals with positive relative learning credit</div></div>
<div class="card"><div class="stat" id="reinforced"></div><div class="label">meal choices made more likely by their update</div></div>
<div class="card"><div class="stat">96 updates</div><div class="label">24,576 recorded world ticks · separate audit passed</div></div></div>
<div class="panel"><label for="candidate">Experimental brain </label><select id="candidate"></select>
<p class="muted" id="detail"></p>
<div class="legend"><span style="--ink:#7fd6ba">Eat carried food, fixed new probes</span><span style="--ink:#91b8ff">Gather adjacent food, fixed new probes</span>
<span style="--ink:#f4cc7d">First meal, zero context</span><span style="--ink:#dd9ae4">First meal, original context held fixed</span></div>
<svg id="chart" viewBox="0 0 960 340" role="img" aria-label="Action probabilities after each of sixteen recorded training updates"></svg>
<p class="note muted">Probability of selecting an action on identical inputs, not chance of survival.
Fixed probes average four orientations and are never training examples. Holding old recurrent context fixed
isolates parameter changes; it does not establish that an agent remembers the episode.</p></div>
<div class="panel scroll"><table><thead><tr><th>Brain</th><th>Meals</th><th>Positive credit</th><th>Reinforced</th><th>Eat probe, before</th><th>Eat probe, after</th></tr></thead><tbody id="table"></tbody></table></div>
<p class="muted">Credit is spread backward through preceding decisions. Immediate credit, policy reinforcement,
generalization and retention are different measurements. This six-brain pool cannot establish a memory defect,
a causal loss-function bottleneck, reliable feeding, or superiority over the live PPO or Laya models.</p>
<p class="note"><a href="../../docs/FEEDING-CREDIT.md">Full method and interpretation</a> · <a href="audit.json">Audit receipt</a> ·
<a href="analysis.json">Summary data</a> · <a href="../culling-pilot-20261006/replay-adjacent.html">Original feeding-trial replay</a></p>
<script id="payload" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent),summary=data.analysis;
const select=document.getElementById('candidate'), svg=document.getElementById('chart');
const pct=x=>(100*x).toFixed(1)+'%';
document.getElementById('credit').textContent=summary.positive_meal_advantages+'/'+summary.ordinary_meals;
document.getElementById('reinforced').textContent=summary.reinforced_meals+'/'+summary.ordinary_meals;
for(const cid of Object.keys(summary.candidates)){const option=document.createElement('option');option.value=cid;option.textContent=cid;select.appendChild(option);}
document.getElementById('table').innerHTML=Object.entries(summary.candidates).map(([cid,s])=>
 `<tr><td>${cid}</td><td>${s.ordinary_meals}</td><td>${s.positive_meal_advantages}</td><td>${s.reinforced_meals}</td><td>${pct(s.canonical.carried.before)}</td><td>${pct(s.canonical.carried.after)}</td></tr>`).join('');
function render(){
 const cid=select.value, rows=data.updates.filter(r=>r.candidate===cid),s=summary.candidates[cid];
 const average=(p,fixture,verb)=>{const vs=Object.entries(p).filter(([k])=>k.startsWith(fixture)).map(([,v])=>v[verb]);return vs.reduce((a,b)=>a+b,0)/vs.length;};
 const series=[['#7fd6ba','carried','eat'],['#91b8ff','adjacent','gather']].map(([color,fixture,verb])=>({color,
   points:[[0,average(rows[0].canonical_before,fixture,verb)],...rows.map(r=>[r.case+1,average(r.canonical_after,fixture,verb)])]}));
 for(const [color,context] of [['#f4cc7d','zero_context'],['#dd9ae4','fixed_context']]){
   const usable=rows.filter(r=>r.anchor_before);if(usable.length)series.push({color,points:[[usable[0].case,usable[0].anchor_before[context].eat],...usable.map(r=>[r.case+1,r.anchor_after[context].eat])]});
 }
 const x=v=>62+v*53.5,y=v=>287-v*245;
 let content='';
 for(const tick of [0,.25,.5,.75,1])content+=`<line x1="62" x2="918" y1="${y(tick)}" y2="${y(tick)}" stroke="#34454c"/><text x="50" y="${y(tick)+5}" text-anchor="end" fill="#aebfc4" font-size="14">${Math.round(tick*100)}%</text>`;
 for(const tick of [0,4,8,12,16])content+=`<text x="${x(tick)}" y="310" text-anchor="middle" fill="#aebfc4" font-size="14">${tick}</text>`;
 content+='<text x="490" y="335" text-anchor="middle" fill="#aebfc4" font-size="14">Recorded continuation updates</text>';
 for(const s of series){content+=`<polyline points="${s.points.map(([a,b])=>x(a)+','+y(b)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="3"/>`;
   for(const [a,b] of s.points)content+=`<circle cx="${x(a)}" cy="${y(b)}" r="3" fill="${s.color}"><title>Update ${a}: ${pct(b)}</title></circle>`;}
 svg.innerHTML=content;
 const retention=s.retention.zero_context;
 document.getElementById('detail').textContent=`${cid}: ${s.reinforced_meals}/${s.ordinary_meals} meal choices reinforced. Fixed carried-food eating probability ${pct(s.canonical.carried.before)} → ${pct(s.canonical.carried.after)}.`+
   (retention?` First-meal zero-context probe ${pct(retention.before)} → ${pct(retention.immediate)} immediately → ${pct(retention.final)} at the end.`:' No ordinary meal anchor.');
}
select.value=Object.keys(summary.candidates)[0];select.addEventListener('change',render);render();
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    folder=args.folder.resolve();root=Path(__file__).resolve().parents[1]
    if not folder.is_relative_to(root/'runs'):parser.error('Use a local evidence directory')
    audit=json.loads((folder/'audit.json').read_text());assert audit['passed']
    complete=json.loads((folder/'complete.json').read_text())
    for name in ('updates.json','analysis.json'):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==complete['evidence_hashes'][name]
    updates=json.loads((folder/'updates.json').read_text())
    compact=[{k:r[k] for k in ('candidate','case','canonical_before','canonical_after','anchor_before','anchor_after')} for r in updates]
    data=dict(analysis=json.loads((folder/'analysis.json').read_text()),updates=compact)
    page=PAGE.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('<','\\u003c'))
    (folder/'report.html').write_text(page,encoding='utf-8')
    print(folder/'report.html')


if __name__=='__main__':main()
