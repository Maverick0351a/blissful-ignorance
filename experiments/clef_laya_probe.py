"""Frozen, synthetic decision probes; never connects to the live game.

48 cases in six template families, three relabelled option orders per case.
These are supplied-rule diagnostics, not independent survival trials.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request

ADAPTER = Path.home() / 'Projects/laya-lab/npu/laya_lite.py'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def adapter():
    spec = importlib.util.spec_from_file_location('local_laya_probe_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    rows = []
    def add(category, state, instruction, options, correct, reason):
        rows.append(dict(id=f'{category}-{1+sum(x["category"] == category for x in rows):02}',
                         category=category, state=state, instruction=instruction,
                         options=options, correct=correct, rationale=reason))
    for i, food in enumerate([0, 1, 2, 0, 3, 0, 1, 0]):
        add('food', dict(fullness=10+i*7, carried_fruit=food, adjacent_ripe_fruit=4+i,
                        rules='Eating one carried fruit adds 25 fullness immediately. Gathering only adds to inventory. No fruit means eating cannot succeed.'),
            'Which offered action increases fullness immediately this turn?',
            {'eat': 'Eat one carried fruit.', 'gather': 'Gather one adjacent fruit.', 'none': 'Neither action can increase fullness this turn.'},
            'eat' if food else 'none', 'Eating requires carried fruit; gathering has no immediate fullness effect.')
    for a, b, da, db in [('amber','green',-25,0), ('amber','green',0,-25),
                         ('blue','red',0,-20), ('blue','red',-20,0),
                         ('pink','purple',-30,-10), ('orange','white',-10,-30),
                         ('black','gold',0,-25), ('black','gold',-25,0)]:
        add('hazard', dict(carried=[a,b], history=[dict(ate=a,fullness_change=25,health_change=da),
                                                 dict(ate=b,fullness_change=25,health_change=db)],
                          rules='Each fruit type always repeats its observed effects in this diagnostic. No other events affected those measurements.'),
            'Which available fruit has evidence that eating it raises fullness without lowering health?',
            {'first': f'Eat {a} fruit.', 'second': f'Eat {b} fruit.', 'none': 'Neither fruit meets both conditions.'},
            'first' if da == 0 else 'second' if db == 0 else 'none',
            'Use the measured health change, including cases where familiar color expectations are reversed.')
    for growth in range(1,5):
        for enough in [False, True]:
            horizon = growth + (3 if enough else 2)
            add('farming', dict(carried_fruit=1, deadline_turns=horizon, growth_wait_turns=growth,
                                rules='Farming consumes the only fruit to make two seeds (1 turn), plants one seed (1 turn), waits the stated growth time, then harvests three edible fruit (1 turn). Every step is sequential. Seeds and unharvested crops are not edible inventory. No other gains, losses or competition occur.'),
                'Which plan leaves the most edible fruit in inventory at the deadline?',
                {'farm': 'Convert, plant, wait for growth, and harvest if time permits.',
                 'keep': 'Keep the initial fruit throughout.', 'eat': 'Eat the initial fruit and wait.'},
                'farm' if enough else 'keep', f'Farming needs {growth+3} turns to yield 3 fruit; keeping retains 1 and eating retains 0.')
    for distance, water, unconscious in [(1,1,True),(2,1,True),(1,0,True),(1,2,True),
                                        (1,1,False),(3,3,True),(1,4,True),(2,0,False)]:
        add('revival', dict(neighbor_distance=distance, carried_water=water, neighbor_unconscious=unconscious,
                           rules='Revive wakes an unconscious neighbor only at distance 1 with at least 1 carried water. Tap attracts attention but never wakes an unconscious neighbor. An awake neighbor needs no revival.'),
            'Which offered action can wake this unconscious neighbor right now?',
            {'revive': 'Revive the neighbor.', 'tap': 'Tap the neighbor.', 'none': 'Neither action can wake an unconscious neighbor in this state.'},
            'revive' if distance == 1 and water >= 1 and unconscious else 'none',
            'All revival preconditions must hold.')
    for i, known in enumerate(['food','water','unknown','food','water','unknown','food','water']):
        food_tone, water_tone = i, (i+1)%10
        heard = food_tone if known == 'food' else water_tone if known == 'water' else (i+2)%10
        add('social', dict(shared_code={str(food_tone):'food nearby', str(water_tone):'water nearby'},
                          heard_tone=heard, rules='This diagnostic supplies an established code. A tone absent from the code has no known meaning. Nothing else is implied.'),
            'What message does the heard tone convey under the supplied code?',
            {'food':'Food is nearby.', 'water':'Water is nearby.', 'unknown':'Its meaning is unknown.'},
            known, 'Lookup the exact tone; do not infer meanings for missing entries.')
    for i, observed in enumerate(['fruit','empty','occluded','occluded','fruit','empty','occluded','fruit']):
        direction = ['north','east','south','west'][i%4]
        add('evidence', dict(target_direction=direction, target_observation=observed,
                            other_visible_tiles=['fruit','empty'],
                            rules='Fruit means fruit is seen at that tile. Empty means no fruit is present there. Occluded means the target cannot be seen; no previous observation of it exists.'),
            'Does the available evidence establish fruit at the target tile?',
            {'yes':'Yes, fruit is present there.', 'no':'No, fruit is absent there.', 'unknown':'The target fruit status is unknown.'},
            {'fruit':'yes','empty':'no','occluded':'unknown'}[observed], 'Respect the target and visibility limit.')
    assert len(rows) == 48 and len({r['id'] for r in rows}) == 48
    return rows


def request_for(case, variant):
    semantic = list(case['options'])
    semantic = semantic[variant:] + semantic[:variant]
    mapping = dict(zip(('a','b','c'), semantic))
    question = dict(type='choice', instructions=case['instruction'],
                    criteria={label:case['options'][key] for label,key in mapping.items()})
    # Identical text string avoids different JSON state renderings in the runtimes.
    state = json.dumps(case['state'], sort_keys=True, ensure_ascii=False)
    return dict(state=state, questions={'action':question}), mapping


def validate_tokens(module, case_rows):
    tok = module.FastTok(module.CKPT / 'tokenizer')
    lengths = []
    for case in case_rows:
        assert len(case['options']) == 3 and case['correct'] in case['options']
        for variant in range(3):
            req, _ = request_for(case, variant)
            q = module.to_internal(req['questions']['action'])
            encoded = lambda text: tok(text, add_special_tokens=False)['input_ids']
            head = encoded('choice question: '+q['ins'])
            option_tokens = [encoded(' '+opt) for opt in module.render_options(q)]
            assert all(len(x) <= 48 for x in option_tokens), 'option truncation'
            option_budget = sum(len(x)+1 for x in option_tokens)
            assert option_budget+len(head) <= 192 and option_budget <= 176, 'head truncation'
            expected = len(head)+option_budget+4+len(encoded(req['state']))
            assert expected <= 512, f'{case["id"]}: state truncation'
            seq, markers = module.build_sequence(tok,req['state'],q,max_len=512,head_max_len=192)
            assert len(seq) == expected and len(markers) == 3
            lengths.append(expected)
    return dict(min=min(lengths),max=max(lengths),requests=len(lengths),truncations=0)


def prepare(output):
    if (output/'preregistration.json').exists():
        raise FileExistsError('Preregistration exists; preserve it and use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    rows = cases()
    tokens = validate_tokens(adapter(), rows)
    write_json(output/'cases.json',rows)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), status='frozen before inference',
                    cases_sha256=digest(output/'cases.json'), runner_sha256=digest(Path(__file__)),
                    adapter_sha256=digest(ADAPTER), cases=48, permutations=3, requests_per_model=144,
                    warmup_requests=3, inference_wall_cap_seconds=600, request_timeout_seconds=30,
                    token_check=tokens, confidence_threshold=0.9,
                    brier_definition='Mean over requests of sum across classes of (p - one_hot_label)^2; no division by class count.',
                    limits='Six repeated template families; permutations are not independent cases. Fixed weights, supplied rules, no live world. No claim of learned competence.')
    write_json(output/'preregistration.json',manifest)
    print(json.dumps(manifest),flush=True)


def http_request(endpoint, payload):
    if not endpoint.startswith('http://127.0.0.1:') or not endpoint.endswith('/v1/systemone'):
        raise ValueError('Only the separate loopback decision endpoint is permitted')
    request = urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=30) as response:
        return json.load(response)


def percentile(values, quantile):
    return sorted(values)[math.ceil(len(values)*quantile)-1]


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['id']].append(record)
    strict = sum(len(rs)==3 and all(r['correct'] for r in rs) for rs in grouped.values())
    unstable = sum(len({r['semantic_choice'] for r in rs})>1 for rs in grouped.values())
    per_category = {}
    for category in sorted({r['category'] for r in records}):
        selected = [r for r in records if r['category']==category]
        case_sets = [rs for rs in grouped.values() if rs[0]['category']==category]
        per_category[category] = dict(correct=sum(r['correct'] for r in selected),requests=len(selected),
                                     all_orders_correct=sum(len(rs)==3 and all(r['correct'] for r in rs) for rs in case_sets),cases=len(case_sets))
    bins = []
    for lo, hi in [(0,.4),(.4,.6),(.6,.8),(.8,.9),(.9,1.000001)]:
        rs = [r for r in records if lo<=r['top_probability']<hi]
        if rs:
            bins.append(dict(range=[lo,min(hi,1)],count=len(rs),accuracy=statistics.mean(r['correct'] for r in rs),
                             mean_top_probability=statistics.mean(r['top_probability'] for r in rs)))
    return dict(requests=len(records),correct=sum(r['correct'] for r in records),
                accuracy=statistics.mean(r['correct'] for r in records),cases=len(grouped),all_orders_correct=strict,
                order_sensitive_cases=unstable,confident_errors=sum(not r['correct'] and r['top_probability']>=.9 for r in records),
                mean_brier=statistics.mean(r['brier'] for r in records),
                latency_median_ms=statistics.median(r['ms'] for r in records),latency_p95_ms=percentile([r['ms'] for r in records],.95),
                categories=per_category,reliability_bins=bins)


def run(output, model, endpoint):
    manifest = json.loads((output/'preregistration.json').read_text(encoding='utf-8'))
    assert digest(output/'cases.json') == manifest['cases_sha256']
    assert digest(Path(__file__)) == manifest['runner_sha256']
    assert digest(ADAPTER) == manifest['adapter_sha256']
    rows = json.loads((output/'cases.json').read_text(encoding='utf-8'))
    validate_tokens(adapter(),rows)
    trace = output/f'{model}-trace.jsonl'
    if trace.exists():
        raise FileExistsError('Do not overwrite an existing run or cherry-pick a retry.')
    started = time.perf_counter()
    if model == 'laya':
        engine = adapter().LayaLite('NPU')
        call = lambda payload: engine.system_one(payload['state'],payload['questions'])
    else:
        call = lambda payload: http_request(endpoint,payload)
    setup_s = time.perf_counter()-started
    warmup = dict(state='The object is a square.',questions={'action':dict(type='choice',instructions='Which shape is stated?',criteria={'a':'circle','b':'square','c':'triangle'})})
    for _ in range(3):
        call(warmup)
    started = time.perf_counter()
    records = []
    try:
        with trace.open('x',encoding='utf-8') as stream:
            for case in rows:
                for variant in range(3):
                    if time.perf_counter()-started > 600:
                        raise TimeoutError('Inference wall budget exceeded')
                    payload,mapping = request_for(case,variant)
                    t0 = time.perf_counter()
                    answer = call(payload)['answers']['action']
                    elapsed = (time.perf_counter()-t0)*1000
                    probs = answer['probabilities']
                    assert set(probs) == set(mapping)
                    assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values())
                    assert abs(sum(probs.values())-1)<.001
                    choice = answer['choice']
                    assert choice in mapping
                    assert probs[choice] >= max(probs.values())-1e-4
                    record = dict(id=case['id'],category=case['category'],variant=variant,
                                  answer=answer,label_mapping=mapping,semantic_choice=mapping[choice],
                                  correct=mapping[choice]==case['correct'],top_probability=probs[choice],
                                  brier=sum((p-int(mapping[k]==case['correct']))**2 for k,p in probs.items()),ms=elapsed)
                    records.append(record)
                    stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
                print(json.dumps(dict(model=model,completed=len(records),last_case=case['id'])),flush=True)
        assert len(records)==144
        result = dict(status='complete',model=model,setup_s=setup_s,inference_s=time.perf_counter()-started,
                      cases_sha256=manifest['cases_sha256'],**summarize(records))
        write_json(output/f'{model}-summary.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write_json(output/f'{model}-failure.json',dict(status='incomplete',error=str(error),completed=len(records)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','laya','clef'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8793/v1/systemone')
    args = parser.parse_args()
    if args.mode=='prepare':
        prepare(args.output)
    else:
        run(args.output,args.mode,args.endpoint)
