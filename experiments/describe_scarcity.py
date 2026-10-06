"""Post hoc action-use diagnostic. It does not change the acceptance gate."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');args=parser.parse_args()
    out=Path(args.directory);rows=json.loads((out/'results.json').read_text(encoding='utf-8'));report={}
    for condition in sorted({r['condition'] for r in rows}):
        counts=Counter();first_eats=[];never_ate=0;zero_crossings=0
        for row in rows:
            if row['condition']!=condition:continue
            first={rid:None for rid in row['metrics']}
            with (out/f"{row['evaluation_seed']}-{condition}-trace.jsonl").open(encoding='utf-8') as handle:
                for line in handle:
                    record=json.loads(line)
                    for rid,m in record['agents'].items():
                        if m['conscious']:counts[m['action']['verb']]+=1
                        if m['nutrition']>0 and first[rid] is None:first[rid]=record['tick']
            first_eats.extend(t for t in first.values() if t is not None)
            never_ate+=sum(t is None for t in first.values())
            zero_crossings+=sum(m['crossings']==0 for m in row['metrics'].values())
        total=sum(counts.values())
        condition_rows=[r for r in rows if r['condition']==condition]
        lives=sum(len(r['metrics']) for r in condition_rows)
        assert len(first_eats)+never_ate==lives
        assert total==sum(m['conscious_decisions'] for r in condition_rows for m in r['metrics'].values())
        report[condition]=dict(lives=lives,conscious_action_counts=dict(counts),
            conscious_action_percent={k:100*v/total for k,v in counts.items()},
            mean_first_nutrition_tick_if_any=sum(first_eats)/max(1,len(first_eats)),
            lives_without_nutrition=never_ate,lives_without_completed_crossing=zero_crossings)
    result=dict(label='Post hoc descriptive trace diagnostic; not a new preregistered hypothesis or causal explanation.',conditions=report)
    (out/'action-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
