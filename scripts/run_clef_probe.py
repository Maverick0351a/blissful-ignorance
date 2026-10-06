"""Bounded local comparison; temporarily unload and restore approved idle Gemma.

Only this wrapper's own benchmark/server process trees can be terminated.
No live-game POSTs, weight updates, downloads or external inference.
"""
import ctypes
import argparse
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'runs/clef-laya-20261005'
ROOT = Path.home()/'AI/clef-flash'
LMS = Path.home()/'.lmstudio/bin/lms.exe'
GEMMA = 'gemma-4-e4b-uncensored-hauhaucs-aggressive-verified'
PYTHON = Path.home()/'Projects/laya-lab/.venv/Scripts/python.exe'
PORT = 8793
HIDDEN = subprocess.CREATE_NO_WINDOW


def save(name, record):
    (OUT/name).write_text(json.dumps(record, indent=2, allow_nan=False),encoding='utf-8')


def log(**record):
    print(json.dumps(record),flush=True)


def command(args, timeout=60):
    # These control calls use only the existing local LM Studio app.
    process = subprocess.Popen([str(x) for x in args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=HIDDEN)
    try:
        out,err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_owned(process)
        raise
    if process.returncode:
        raise RuntimeError(f'Local command failed ({process.returncode}): '+err.decode(errors='replace')[-1000:])
    return out.decode(encoding='utf-8-sig',errors='replace')


def stop_owned(process):
    if process is not None and process.poll() is None:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15,creationflags=HIDDEN)
        process.wait(timeout=15)


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
        (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


kernel = ctypes.WinDLL('kernel32',use_last_error=True)
psapi = ctypes.WinDLL('psapi',use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]


class ProcessEntry(ctypes.Structure):
    _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('pid',wintypes.DWORD),
              ('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),
              ('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),
              ('exe',wintypes.WCHAR*260)]


kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
for name in ('Process32FirstW','Process32NextW'):
    getattr(kernel,name).argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessEntry)]


def tree_memory(process):
    """Include the real Python child behind a Windows venv launcher."""
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:
        raise OSError('Process snapshot unavailable; do not report parent-only memory.')
    links=[]
    try:
        entry=ProcessEntry(); entry.dwSize=ctypes.sizeof(entry)
        more=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while more:
            links.append((int(entry.pid),int(entry.parent)))
            more=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    pids={process.pid}
    while True:
        expanded=pids|{pid for pid,parent in links if parent in pids}
        if expanded==pids:
            break
        pids=expanded
    return [dict(record,root_pid=process.pid) for pid in sorted(pids) if (record:=memory(pid))]


def memory(pid):
    handle = kernel.OpenProcess(0x0400|0x0010,False,pid)
    if not handle:
        return None
    try:
        data=MemoryCounters(); data.cb=ctypes.sizeof(data)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(data),data.cb):
            return None
        return dict(pid=pid,working_bytes=data.WorkingSetSize,peak_working_bytes=data.PeakWorkingSetSize,
                    private_bytes=data.PrivateUsage)
    finally:
        kernel.CloseHandle(handle)


def gpu():
    values = command(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],timeout=10)
    total,used=[int(x.strip()) for x in values.splitlines()[0].split(',')]
    return dict(total_mib=total,used_mib=used,free_mib=total-used)


samples=[]
def sample(stage, processes):
    row=dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,gpu=gpu(),
             processes=[record for process in processes if process is not None for record in tree_memory(process)])
    samples.append(row)
    with (OUT/'resource-samples.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')


def get_health():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:
        return json.load(response)


def run_worker(model, env, server=None):
    with (OUT/f'{model}-worker.log').open('x',encoding='utf-8') as stream:
        process=subprocess.Popen([str(PYTHON),'-u',str(REPO/'experiments/clef_laya_probe.py'),model,'--output',str(OUT)],
                                 stdout=stream,stderr=subprocess.STDOUT,cwd=REPO,env=env,creationflags=HIDDEN)
        save(f'{model}-worker-process.json',dict(pid=process.pid,started_at=datetime.now(timezone.utc).isoformat()))
        started=time.monotonic(); last_progress=started; last_size=0; last_report=started
        try:
            while process.poll() is None:
                now=time.monotonic()
                trace=OUT/f'{model}-trace.jsonl'
                size=trace.stat().st_size if trace.exists() else 0
                if size!=last_size:
                    last_progress=now; last_size=size
                if now-started>780 or (last_size and now-last_progress>35) or (not last_size and now-started>180):
                    raise TimeoutError(f'{model} exceeded its load, progress or total watchdog budget')
                sample(model,[process,server])
                if now-last_report>=20:
                    log(stage=model,elapsed_s=round(now-started),trace_bytes=size)
                    last_report=now
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'{model} worker failed; see its preserved log')
        finally:
            stop_owned(process)
    result=json.loads((OUT/f'{model}-summary.json').read_text())
    log(stage=model,status='complete',correct=result['correct'],requests=result['requests'])


def main():
    if '--approved-temporary-gemma-unload' not in sys.argv:
        raise RuntimeError('The explicit temporary-unload option is required.')
    if (OUT/'orchestration-result.json').exists():
        raise FileExistsError('Preserve the prior run; do not reuse its result directory.')
    state=json.loads(command([LMS,'ps','--json']))
    gemma=next((m for m in state if m.get('identifier')==GEMMA),None)
    if not gemma or gemma['status']!='idle' or gemma.get('queued',0):
        raise RuntimeError('Approved Gemma is not idle; leave local applications alone.')
    save('gemma-at-unload.json',gemma)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',PORT))
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','HF_','HUGGINGFACE_'))}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    server=None; server_log=None; unload_requested=False
    outcome=dict(status='incomplete')
    try:
        # Set before dispatch so a partial/uncertain unload still triggers restore.
        unload_requested=True
        command([LMS,'unload',GEMMA],timeout=30)
        log(stage='gemma',status='temporarily_unloaded')
        available=gpu(); save('gpu-after-unload.json',available)
        if available['free_mib']<8192:
            raise RuntimeError('Fewer than 8 GiB GPU memory free after the approved unload; no Clef load attempted.')
        run_worker('laya',env)
        args=[str(ROOT/'runtime-b11429/llama-server.exe'),'-m',str(ROOT/'Clef-Flash-Q4_K_M.gguf'),
              '--alias','clef-flash-q4-local-probe','--host','127.0.0.1','--port',str(PORT),
              '-c','1024','-b','512','-ub','512','-np','1','-ngl','99','--fit','off','-t','4','-tb','4',
              '--offline','--no-webui','--no-agent','--no-ui-mcp-proxy',
              '--cors-origins',f'http://127.0.0.1:{PORT}','--no-cors-credentials']
        save('clef-launch.json',dict(args=args))
        server_log=(OUT/'clef-server.log').open('x',encoding='utf-8')
        start=time.monotonic()
        server=subprocess.Popen(args,stdout=server_log,stderr=subprocess.STDOUT,cwd=ROOT/'runtime-b11429',
                                env=env,creationflags=HIDDEN)
        save('clef-server-process.json',dict(pid=server.pid,started_at=datetime.now(timezone.utc).isoformat()))
        while True:
            if server.poll() is not None:
                raise RuntimeError('Clef server exited during load; see preserved server log.')
            if time.monotonic()-start>180:
                raise TimeoutError('Clef load exceeded 180 seconds.')
            sample('clef_load',[server])
            try:
                health=get_health()
                if health.get('status')=='ok':
                    break
            except (OSError,ValueError):
                pass
            time.sleep(2)
        load_seconds=time.monotonic()-start
        save('clef-health.json',dict(health=health,cold_load_to_health_seconds=load_seconds))
        log(stage='clef',status='loaded',cold_load_seconds=round(load_seconds,2))
        run_worker('clef',env,server)
        outcome=dict(status='complete',clef_cold_load_seconds=load_seconds)
    except BaseException as error:
        outcome['error']=f'{type(error).__name__}: {error}'
        log(stage='comparison',**outcome)
    finally:
        stop_owned(server)
        if server_log:
            server_log.close()
        save('resource-summary.json',dict(samples=len(samples),
             global_gpu_peak_mib=max((x['gpu']['used_mib'] for x in samples),default=None),
             stages={stage:dict(global_gpu_peak_mib=max(x['gpu']['used_mib'] for x in samples if x['stage']==stage),
                               peak_process_working_bytes=max((p['peak_working_bytes'] for x in samples if x['stage']==stage for p in x['processes']),default=0),
                               peak_process_private_bytes=max((p['private_bytes'] for x in samples if x['stage']==stage for p in x['processes']),default=0))
                     for stage in sorted({x['stage'] for x in samples})}))
        if unload_requested:
            try:
                current=json.loads(command([LMS,'ps','--json']))
                if not any(m.get('identifier')==GEMMA for m in current):
                    command([LMS,'load',gemma['modelKey'],'--identifier',GEMMA,'--context-length',str(gemma['contextLength']),
                             '--parallel',str(gemma['parallel']),'--yes'],timeout=180)
                restored=json.loads(command([LMS,'ps','--json']))
                save('gemma-restored.json',restored)
                restored_gemma=next(m for m in restored if m.get('identifier')==GEMMA)
                assert all(restored_gemma[k]==gemma[k] for k in ('modelKey','contextLength','parallel'))
                outcome['gemma_restored']=True
                log(stage='gemma',status='restored')
            except BaseException as error:
                outcome['gemma_restored']=False
                outcome['restore_error']=str(error)
                log(stage='gemma',status='RESTORE_FAILED',error=str(error))
        save('orchestration-result.json',outcome)
    print(json.dumps(outcome),flush=True)
    if outcome['status']!='complete' or not outcome.get('gemma_restored'):
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--approved-temporary-gemma-unload',action='store_true',required=True)
    args=parser.parse_args()
    OUT=args.output.resolve()
    main()
