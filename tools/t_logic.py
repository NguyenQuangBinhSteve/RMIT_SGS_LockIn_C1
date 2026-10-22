import json, os, glob, tempfile, ast
import sys
nb = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks', 'RMIT_SGS_Lock-In_C1.ipynb')))
src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
tree = ast.parse(src)
def fn(name): return ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
g = {'os': os, 'glob': glob, 'HEAD': 4, 'TAIL': 2}
class Tok:
    def __call__(self, t): return {'input_ids': [[0] + list(range(1, n + 1)) + [9] for n in map(len, t)]}
g['tok'] = Tok()
for n in ('find_model', 'can_continue', 'encode'): exec(fn(n), g)
# mô hình chỉ có pytorch_model.bin + thư mục onnx có config
r = tempfile.mkdtemp()
for p in ['m/a/config.json', 'm/a/pytorch_model.bin', 'm/a/onnx/config.json', 'm/a/onnx/model.safetensors', 'm/.cache/x/config.json', 'm/.cache/x/model.safetensors', 'c/train.csv']:
    os.makedirs(os.path.dirname(f'{r}/{p}'), exist_ok=True); open(f'{r}/{p}', 'w').close()
assert g['find_model'](r) == f'{r}/m/a'
os.remove(f'{r}/m/a/pytorch_model.bin'); open(f'{r}/m/a/model.safetensors', 'w').close()
assert g['find_model'](r) == f'{r}/m/a'
os.remove(f'{r}/m/a/model.safetensors')
try: g['find_model'](r); assert 0
except AssertionError as e: assert e.args == ([],) or e.args[0] == []
cc = g['can_continue']
assert cc(10, [20, 20, 20], 55) and not cc(40, [20, 20], 55) and cc(35, [20], 55)
e = g['encode'](['a' * 3, 'a' * 100])
assert len(e[0]) == 5
assert e[1] == [0,1,2,3,4] + [99,100,9]
print('ok')

if 'def save_run' in src:
    import numpy as np, pandas as pd, time, tempfile
    from sklearn.metrics import confusion_matrix, f1_score
    rng = np.random.default_rng(0)
    n, nt = 60, 10
    y = np.arange(n) % 3
    train = pd.DataFrame({'y': y, 'text': ['x' * (i * 20) for i in range(n)], 'language': ['en', 'vi'] * (n // 2)})
    test = pd.DataFrame({'id': range(nt)})
    G = dict(np=np, pd=pd, json=json, time=time, f1_score=f1_score, confusion_matrix=confusion_matrix, train=train, test=test,
             LABELS=['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'], SMOKE=True, K=3, T0=time.time(), HF_NAME='h', MIRROR='m',
             EPOCHS=1, LR=1e-5, GC=False, FP16=True, PARALLEL=False, HEAD=1, TAIL=1, DATA='')
    exec(fn('save_run'), G)
    G.update(oof=np.zeros((n, 3)), pte=np.zeros((nt, 3)), mask=np.zeros(n, bool), fold_f1=[], fold_min=[])
    os.chdir(tempfile.mkdtemp())
    for f, vi in enumerate([np.arange(0, 20), np.arange(20, 40)]):
        G['oof'][vi, y[vi]] = 1.0
        G['pte'] += np.eye(3)[rng.integers(0, 3, nt)]
        G['mask'][vi] = True
        G['fold_f1'].append(1.0); G['fold_min'].append(1.0)
        G['save_run']()
        info = json.load(open('run_info.json'))
        assert info['folds_done'] == f + 1 and info['stopped_early'] and info['oof_tuned'] == 1.0
        assert np.load('oof.npy').shape == (n, 3) and np.allclose(np.load('test_probs.npy').sum(1), 1)
        assert len(pd.read_csv('submission.csv')) == nt
    G['save_run'](True)
    print('save_run ok')


if True:
    import subprocess, types
    d = tempfile.mkdtemp()
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, f'{here}/mk_nb.py', f'{d}/par.ipynb', '--model', 'base', '--parallel-folds', '--k', '4'], check=True)
    psrc = '\n'.join(''.join(c['source']) for c in json.load(open(f'{d}/par.ipynb'))['cells'] if c['cell_type'] == 'code')
    ptree = ast.parse(psrc)
    def pfn(name): return ast.get_source_segment(psrc, next(n for n in ptree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    worker = next(n.value.value for n in ptree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'WORKER_SRC')
    ast.parse(worker)
    nf = lambda q, e, m, b: None
    G2 = dict(can_continue=cc)
    exec(pfn('next_fold'), G2)
    nf = G2['next_fold']
    q = [0, 1, 2, 3]
    assert nf(q, 10, [], 55) == 0 and nf(q, 10, [20], 55) == 1 and nf(q, 40, [20], 55) is None and q == [2, 3]
    assert nf([], 0, [], 55) is None
    # chạy run_parallel với worker giả: 2 GPU, 4 fold; fold lỗi; dừng sớm theo ngân sách
    fake = (
        "import sys, os, json, numpy as np\n"
        "f = int(sys.argv[1]); job = json.load(open('pf/job.json'))\n"
        "open(f'pf/gpu_{f}.txt', 'w').write(os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "if str(f) == os.environ.get('FAIL'): print('boom'); sys.exit(3)\n"
        "vi = job['splits'][f][1]\n"
        "np.save(f'pf/oof_{f}.npy', np.eye(3)[[i % 3 for i in vi]]); np.save(f'pf/test_{f}.npy', np.ones((10, 3)) / 3)\n")
    import numpy as np, pandas as pd
    from sklearn.metrics import f1_score
    n = 60
    tr = pd.DataFrame({'y': np.arange(n) % 3})
    class DS:
        def save_to_disk(self, p): pass
    def run(budget, fail=None):
        os.chdir(tempfile.mkdtemp())
        if fail is None: os.environ.pop('FAIL', None)
        else: os.environ['FAIL'] = fail
        G = dict(os=os, json=json, time=time, subprocess=subprocess, sys=sys, np=np, f1_score=f1_score, train=tr,
                 torch=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 2)),
                 ds_all=DS(), ds_te=DS(), class_w=types.SimpleNamespace(tolist=lambda: [1, 1, 1]), MODEL='m', EPOCHS=1, LR=1e-5,
                 GC=False, FP16=True, SEED=1, T0=time.time(), TIME_BUDGET_MIN=budget, WORKER_SRC=fake,
                 oof=np.zeros((n, 3)), pte=np.zeros((10, 3)), mask=np.zeros(n, bool), fold_f1=[], fold_min=[], can_continue=cc,
                 next_fold=nf, save_run=lambda final=False: None)
        exec(pfn('add_fold'), G)
        exec(pfn('run_parallel'), G)
        splits = [(np.array([i for i in range(n) if i // 15 != f]), np.arange(f * 15, f * 15 + 15)) for f in range(4)]
        try:
            G['run_parallel'](splits)
        finally:
            G['gpus'] = sorted(open(f'pf/gpu_{f}.txt').read() for f in range(4) if os.path.exists(f'pf/gpu_{f}.txt'))
        return G
    G = run(1000)
    assert G['mask'].all() and G['gpus'] == ['0', '0', '1', '1'] and len(G['fold_min']) == 4 and min(G['fold_f1']) == 1.0
    assert np.allclose(G['pte'], 4 / 3)
    try: run(1000, fail='2'); assert 0
    except RuntimeError as e: assert 'fold 2' in str(e)
    G = run(0.0001)
    assert len(G['fold_min']) == 2 and G['mask'].sum() == 30
    print('run_parallel ok')
