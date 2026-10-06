"""Standalone offline viewer for audited paired update effects."""
import argparse
import hashlib
import json
from pathlib import Path

PAGE=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Godhood Trials · Comparing learning updates</title>
<style>:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10191d;color:#e3ecee}body{max-width:1120px;margin:auto;padding:28px}
h1{font-size:clamp(26px,5vw,40px)}.muted{color:#afc2ca;line-height:1.6}.eyebrow{letter-spacing:.12em;color:#93bac5;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.card,.panel{border:1px solid #34454c;border-radius:12px;background:#18252b;padding:20px}.panel{margin:20px 0}.stat{font-size:28px;font-weight:650}.label,.note{font-size:13px;color:#b4c5cb}
select{background:#10191d;border:1px solid #8bb3c1;border-radius:6px;color:white;padding:8px;font:inherit}svg{display:block;width:100%;height:auto}.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px 10px;border-bottom:1px solid #34454c;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#8dd1eb}
.legend{display:flex;gap:24px;flex-wrap:wrap}.native{color:#8ab8ff}.detached{color:#84d7b3}@media(max-width:680px){body{padding:16px}.cards{grid-template-columns:1fr}.panel{padding:12px}}
</style><div class="eyebrow">GODHOOD TRIALS / GT-01 / OCTOBER 6, 2026</div><h1>Comparing two learning updates</h1>
<p class="muted">Does preventing the future-reward estimator from changing the shared policy network improve eating choices?
Each comparison begins with identical weights, optimizer history and experience. Every experimental copy receives just one update.
The next comparison resets to the next original checkpoint; these changes do not accumulate into a trained experimental agent.</p>
<div class="cards"><div class="card"><div class="stat" id="effect"></div><div class="label">extra eating-probability gain per matched update</div></div>
<div class="card"><div class="stat" id="brains"></div><div class="label">brains with a positive mean paired effect</div></div>
<div class="card"><div class="stat" id="screen"></div><div class="label">predeclared diagnostic screen; GT-01 remains open</div></div></div>
<div class="panel"><label for="candidate">Experimental brain </label><select id="candidate"></select><p class="muted" id="detail"></p>
<div class="legend"><span class="native">● Standard PPO gain</span><span class="detached">● Value-gradient-blocked gain</span></div>
<svg id="chart" viewBox="0 0 960 350" role="img" aria-label="Paired eating-probability gains at sixteen original update boundaries"></svg>
<p class="note">Vertical units are percentage points of action probability, not survival success. Each point compares an update to its own unchanged starting brain.
Probes average four fixed carried-food scenes previously inspected in the feeding diagnostic. They were never training examples.</p></div>
<div class="panel scroll"><table><thead><tr><th>Brain</th><th>Standard gain (pp)</th><th>Blocked gain (pp)</th><th>Extra gain (pp)</th><th>Meal choices reinforced, standard / blocked</th></tr></thead><tbody id="table"></tbody></table></div>
<p class="muted">The value head still learns. Rewards, actions, targets and optimizer history are unchanged. Changes to the shared gradient-clipping scale are included in the intervention.
This measures immediate effects on the original training distribution; it does not establish sustained feeding, memory quality or long-term behavior.</p>
<p class="note"><a href="../../docs/VALUE-INTERFERENCE.md">Full method and findings</a> · <a href="audit.json">Audit receipt</a> · <a href="analysis.json">Summary data</a></p>
<script id="payload" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent),summary=data.analysis,select=document.getElementById('candidate');
const pp=x=>(x>=0?'+':'')+(x*100).toFixed(4),average=p=>{const a=Object.entries(p).filter(([k])=>k.startsWith('carried')).map(([,v])=>v.eat);return a.reduce((x,y)=>x+y,0)/a.length;};
document.getElementById('effect').textContent=pp(summary.mean_paired_carried_difference)+' pp';
document.getElementById('brains').textContent=summary.positive_brains+'/6';document.getElementById('screen').textContent=summary.diagnostic_promising?'Screen met':'Screen failed';
for(const cid of Object.keys(summary.candidates)){const option=document.createElement('option');option.value=cid;option.textContent=cid;select.appendChild(option);}
document.getElementById('table').innerHTML=Object.entries(summary.candidates).map(([cid,s])=>{const f=s.fixtures.carried;return `<tr><td>${cid}</td><td>${pp(f.mean_standard_gain)}</td><td>${pp(f.mean_detached_gain)}</td><td>${pp(f.mean_paired_difference)}</td><td>${s.reinforced.standard} / ${s.reinforced.detached} of ${s.meals}</td></tr>`;}).join('');
function render(){
 const cid=select.value,rows=data.pairs.filter(r=>r.candidate===cid),s=summary.candidates[cid];
 const series=[['standard','#8ab8ff'],['detached','#84d7b3']].map(([arm,color])=>({color,points:rows.map(r=>[r.case+1,100*(average(r['probes_'+arm])-average(r.probes_before))])}));
 const all=series.flatMap(s=>s.points.map(p=>p[1])),low=Math.min(0,...all),high=Math.max(0,...all),pad=Math.max(.02,(high-low)*.1),lo=low-pad,hi=high+pad;
 const x=v=>75+(v-1)*55,y=v=>290-240*(v-lo)/(hi-lo);let content='';
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;content+=`<line x1="75" x2="900" y1="${y(v)}" y2="${y(v)}" stroke="#34454c"/><text x="65" y="${y(v)+4}" text-anchor="end" fill="#afc2ca" font-size="14">${v.toFixed(3)}</text>`;}
 content+=`<line x1="75" x2="900" y1="${y(0)}" y2="${y(0)}" stroke="#8a9aa0" stroke-dasharray="5 4"/>`;
 for(const v of [1,4,8,12,16])content+=`<text x="${x(v)}" y="315" text-anchor="middle" fill="#afc2ca" font-size="14">${v}</text>`;
 content+='<text x="480" y="344" text-anchor="middle" fill="#afc2ca" font-size="14">Original update boundary (separate matched copy at every point)</text>';
 for(const s of series){content+=`<polyline points="${s.points.map(([a,b])=>x(a)+','+y(b)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="3"/>`;for(const [a,b] of s.points)content+=`<circle cx="${x(a)}" cy="${y(b)}" r="3" fill="${s.color}"><title>Boundary ${a}: ${b.toFixed(4)} percentage points</title></circle>`;}
 document.getElementById('chart').innerHTML=content;document.getElementById('detail').textContent=`${cid}: extra eating gain ${pp(s.fixtures.carried.mean_paired_difference)} pp per update; extra gathering gain ${pp(s.fixtures.adjacent.mean_paired_difference)} pp. ${s.fixtures.carried.beneficial_updates}/16 updates favor the blocked-gradient copy on carried-food probes.`;
}
select.value=Object.keys(summary.candidates)[0];select.addEventListener('change',render);render();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    folder=args.folder.resolve();root=Path(__file__).resolve().parents[1]
    if not folder.is_relative_to(root/'runs') or folder==root/'runs':parser.error('Use a local evidence folder')
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    assert read('audit.json')['passed'];receipt=read('complete.json')
    for name in ('pairs.json','analysis.json'):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==receipt['evidence_hashes'][name]
    pairs=[{k:r[k] for k in ('candidate','case','probes_before','probes_standard','probes_detached')} for r in read('pairs.json')]
    payload=json.dumps(dict(analysis=read('analysis.json'),pairs=pairs),separators=(',',':')).replace('<','\\u003c')
    (folder/'report.html').write_text(PAGE.replace('__DATA__',payload),encoding='utf-8');print(folder/'report.html')
