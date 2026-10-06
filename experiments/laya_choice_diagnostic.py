"""Follow-up prompted by architecture screen: counterbalance labels and ordering.

Frozen inference only. Existing local adapter required; no model downloads.
This is a prompt diagnostic, not another learning run or tuned replacement result.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    spec=importlib.util.spec_from_file_location('adapter',args.adapter)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    model=adapter.LayaLite('NPU')
    cases=[]
    for good in ('A','B'):
        for order in ('AB','BA'):
            for words in (False,True):
                state={'recent_oldest_first':[{'choice': a, 'outcome':
                       ('benefit' if a==good else 'harm') if words else (1.0 if a==good else -1.52)}
                       for a in 'ABABABAB']}
                q={'action':{'type':'choice','instructions':
                    'Choose food A or B to maximize outcome. Positive is good, negative is bad. Associations can change. Use recent outcomes.',
                    'criteria':{a:'consume '+a for a in order}}}
                cases.append({'good':good,'option_order':order,'word_outcomes':words,
                              'answer':model.system_one(state,q)['answers']['action']})
    Path(args.output).write_text(json.dumps(cases,indent=2),encoding='utf-8')
    for c in cases: print(json.dumps(c),flush=True)


if __name__=='__main__':main()
