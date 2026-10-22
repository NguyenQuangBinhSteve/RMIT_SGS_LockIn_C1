import argparse, json, os

MODELS = {
    'base': ('FacebookAI/xlm-roberta-base', 'aisuko/xlm-roberta-base/transformers/default/1', 2e-5, 55),
    'large': ('FacebookAI/xlm-roberta-large', 'aisuko/xlm-roberta-large/transformers/default/1', 1e-5, 140),
    'mmbert': ('jhu-clsp/mmBERT-base', 'aisuko/mmbert-base/transformers/default/1', 3e-5, 55),
}
SLUGS = {
    'main': ('qucanhtrnnguyn/rmit-sgs-lock-in-c1', 'RMIT_SGS_Lock-In_C1', 'RMIT_SGS_Lock-In_C1.ipynb'),
    'exp': ('qucanhtrnnguyn/rmit-sgs-lockinexp-c1', 'RMIT_SGS_LockInExp_C1', 'RMIT_SGS_LockInExp_C1.ipynb'),
    'probe': ('qucanhtrnnguyn/rmit-sgs-lockinprobe-c1', 'RMIT_SGS_LockInProbe_C1', 'RMIT_SGS_LockInProbe_C1.ipynb'),
}

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--model', choices=MODELS, required=True)
ap.add_argument('--slug', choices=SLUGS, default='main')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--lr', type=float)
ap.add_argument('--head', type=int, default=384)
ap.add_argument('--tail', type=int, default=126)
ap.add_argument('--budget', type=int)
ap.add_argument('--gc', action=argparse.BooleanOptionalAction)
ap.add_argument('--no-fp16', action='store_true')
ap.add_argument('--k', type=int, default=3)
ap.add_argument('--parallel-folds', action='store_true')
a = ap.parse_args()
hf, source, lr0, budget0 = MODELS[a.model]
lr, budget = a.lr or lr0, a.budget or budget0
kid, title, code_file = SLUGS[a.slug]
gc = a.model == 'large' if a.gc is None else a.gc
mirror = source.split('/transformers')[0]

md = f"""# RMIT_SGS_Lock-In_C1
**Model: {hf} (mirror {mirror})**

K={a.k}, epochs={a.epochs}, LR={lr}{', parallel folds' if a.parallel_folds else ''}{' (smoke run)' if a.smoke else ''}

Fine-tune một model được phép cho phân loại 3 lớp BENIGN / JAILBREAK / PROMPT_INJECTION.
Setup: Add Input -> competition `rmit-hackathon-2026` + Models -> `{mirror}`. Accelerator: GPU T4 x2, Internet off.
Quy trình: StratifiedGroupKFold theo prefix 60 ký tự, OOF macro F1, test = trung bình xác suất các fold đã xong."""
cfg = repr((hf, mirror, a.k, a.epochs, lr, a.head, a.tail, budget, a.smoke, gc, not a.no_fp16, a.parallel_folds))

FOLD_LIB = '''# Softmax theo hàng.
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

# Trainer dùng cross-entropy có trọng số lớp để bù lớp PROMPT_INJECTION hiếm.
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = out.logits.float()
        loss = torch.nn.functional.cross_entropy(logits, inputs['labels'], weight=class_w.to(logits.device))
        return (loss, out) if return_outputs else loss

# Dự đoán xác suất cho dataset theo thứ tự độ dài tăng dần rồi trả về đúng thứ tự ban đầu.
def predict_probs(trainer, ds):
    order = np.argsort([len(x) for x in ds['input_ids']])
    p = softmax(trainer.predict(ds.select(order.tolist())).predictions)
    out = np.empty_like(p)
    out[order] = p
    return out

# Huấn luyện một fold, trả về xác suất val, xác suất test và số phút train.
def train_fold(ds_tr, ds_va, ds_te):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    steps = math.ceil(len(ds_tr) / (8 * 2 * max(1, torch.cuda.device_count()))) * EPOCHS
    kw = dict(
        output_dir='/tmp/out', num_train_epochs=EPOCHS, learning_rate=LR, warmup_steps=int(0.1 * steps), weight_decay=0.01,
        per_device_train_batch_size=8, gradient_accumulation_steps=2, per_device_eval_batch_size=64,
        gradient_checkpointing=GC, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=FP16, group_by_length=True, seed=SEED, save_strategy='no', report_to='none', logging_steps=50, disable_tqdm=True)
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in ta_params})
    trainer = WeightedTrainer(model=model, args=args, train_dataset=ds_tr,
                              processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    t_train = time.time()
    trainer.train()
    train_min = (time.time() - t_train) / 60
    pv, pt = predict_probs(trainer, ds_va), predict_probs(trainer, ds_te)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return pv, pt, train_min
'''

WORKER_PRE = '''import gc, inspect, json, math, os, sys, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np, torch
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

job = json.load(open('pf/job.json'))
MODEL, EPOCHS, LR, GC, FP16, SEED = (job[k] for k in ('model', 'epochs', 'lr', 'gc', 'fp16', 'seed'))
class_w = torch.tensor(job['class_w'], dtype=torch.float)
tok = AutoTokenizer.from_pretrained(MODEL)
ta_params = inspect.signature(TrainingArguments).parameters

'''

WORKER_MAIN = '''
f = int(sys.argv[1])
print('fold', f, 'CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'), 'device', torch.cuda.current_device(),
      torch.cuda.get_device_name(0), flush=True)
ds_all, ds_te = load_from_disk('pf/ds_all'), load_from_disk('pf/ds_te')
ti, vi = job['splits'][f]
pv, pt, train_min = train_fold(ds_all.select(ti), ds_all.select(vi), ds_te)
np.save(f'pf/oof_{f}.npy', pv)
np.save(f'pf/test_{f}.npy', pt)
print(f'fold {f} train_min {train_min:.1f}', flush=True)
'''
WORKER = WORKER_PRE + FOLD_LIB + WORKER_MAIN
assert "'''" not in WORKER and chr(92) not in WORKER

PARALLEL_CODE = '''
WORKER_SRC = r\'\'\'@WORKER@\'\'\'

# Lấy fold kế tiếp trong hàng đợi nếu chưa có fold nào xong hoặc còn đủ ngân sách; trả về None nếu phải dừng.
def next_fold(queue, elapsed, fold_mins, budget):
    if queue and (not fold_mins or can_continue(elapsed, fold_mins, budget)):
        return queue.pop(0)

# Chạy các fold song song, mỗi GPU một tiến trình fold_worker.py; worker lỗi thì in cuối log rồi raise.
def run_parallel(splits):
    os.makedirs('pf', exist_ok=True)
    ds_all.save_to_disk('pf/ds_all')
    ds_te.save_to_disk('pf/ds_te')
    job = dict(model=MODEL, epochs=EPOCHS, lr=LR, gc=GC, fp16=FP16, seed=SEED, class_w=class_w.tolist(),
               splits=[[ti.tolist(), vi.tolist()] for ti, vi in splits])
    json.dump(job, open('pf/job.json', 'w'))
    open('fold_worker.py', 'w').write(WORKER_SRC)
    os.system('free -g')
    queue, running = list(range(len(splits))), {}
    while queue or running:
        for g in range(max(1, torch.cuda.device_count())):
            if g in running or not queue:
                continue
            f = next_fold(queue, (time.time() - T0) / 60, fold_min, TIME_BUDGET_MIN)
            if f is None:
                print(f'STOP EARLY after {len(fold_min)} folds', flush=True)
                queue.clear()
            else:
                log = open(f'pf/fold_{f}.log', 'w')
                p = subprocess.Popen([sys.executable, 'fold_worker.py', str(f)], stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(g)})
                running[g] = (f, time.time(), p)
        time.sleep(2)
        for g, (f, t_fold, p) in list(running.items()):
            if p.poll() is None:
                continue
            del running[g]
            if p.returncode:
                print(''.join(open(f'pf/fold_{f}.log').readlines()[-40:]), flush=True)
                for _, _, q in running.values():
                    q.kill()
                raise RuntimeError(f'fold {f} worker failed, returncode {p.returncode}')
            add_fold(f, splits[f][1], np.load(f'pf/oof_{f}.npy'), np.load(f'pf/test_{f}.npy'), t_fold)
'''

CORE = "# Cho phép chạy fold kế tiếp nếu thời gian đã dùng cộng thời gian fold trung bình (phút) chưa vượt ngân sách.\ndef can_continue(elapsed, fold_mins, budget):\n    return elapsed + sum(fold_mins) / len(fold_mins) <= budget\n\n# Từ các fold đã xong: tinh chỉnh trọng số lớp trên OOF, ghi oof.npy, test_probs.npy, submission.csv, run_info.json; final=True thì in báo cáo đầy đủ và kiểm tra định dạng.\ndef save_run(final=False):\n    p = pte / len(fold_min)\n    yv, ov = train.y.values[mask], oof[mask]\n    grid = np.arange(0.5, 2.01, 0.25)\n    best = max(((a, b) for a in grid for b in grid), key=lambda w: f1_score(yv, (ov * [1, *w]).argmax(1), average='macro'))\n    w = np.array([1, *best])\n    pred = (ov * w).argmax(1)\n    oof_raw, oof_tuned = f1_score(yv, ov.argmax(1), average='macro'), f1_score(yv, pred, average='macro')\n    print('OOF macro F1 raw', oof_raw, '| weights', w, 'tuned', oof_tuned, '| folds done', len(fold_min), 'of', K, flush=True)\n    sub = pd.DataFrame({'id': test.id, 'label': [LABELS[i] for i in (p * w).argmax(1)]})\n    np.save('oof.npy', oof)\n    np.save('test_probs.npy', p)\n    sub.to_csv('submission.csv', index=False)\n    json.dump(dict(model=HF_NAME, mirror=MIRROR, K=K, folds_done=len(fold_min), stopped_early=len(fold_min) < K,\n                   epochs=EPOCHS, lr=LR, gc=GC, fp16=FP16, parallel=PARALLEL, head=HEAD, tail=TAIL, oof_raw=oof_raw, oof_tuned=oof_tuned, weights=w.tolist(),\n                   fold_f1=fold_f1, fold_min=fold_min, total_min=(time.time() - T0) / 60,\n                   test_pred_dist=sub.label.value_counts().to_dict()), open('run_info.json', 'w'), indent=1)\n    if not final:\n        return\n    print(confusion_matrix(yv, pred))\n    dv = train[mask].assign(p=pred)\n    bucket = pd.cut(dv.text.str.len(), [-1, 400, 800, 10**9], labels=['0-400', '400-800', '800+'])\n    for name, key in [('length', bucket), ('language', dv.language)]:\n        for k, d in dv.groupby(key, observed=True):\n            print(name, k, len(d), round(f1_score(d.y, d.p, average='macro'), 4))\n    print(sub.label.value_counts(normalize=True))\n    if not SMOKE:\n        ref = pd.read_csv(f'{DATA}/sample_submission.csv')\n        assert len(sub) == 6000 and set(sub.id) == set(ref.id) and set(sub.label) <= set(LABELS)\n\n# Gộp kết quả một fold vào OOF và test, in dòng F1 rồi ghi lại toàn bộ output.\ndef add_fold(f, vi, pv, pt, t_fold, note=''):\n    oof[vi] = pv\n    pte[:] += pt\n    mask[vi] = True\n    fold_f1.append(f1_score(train.y[vi], pv.argmax(1), average='macro'))\n    fold_min.append((time.time() - t_fold) / 60)\n    print(f'fold {f} macro F1 {fold_f1[-1]:.4f}{note} | fold_min {fold_min[-1]:.1f}', flush=True)\n    save_run()\n\n"
SEQ = '''for f, (ti, vi) in enumerate(splits):
    t_fold = time.time()
    pv, pt, train_min = train_fold(ds_all.select(ti), ds_all.select(vi), ds_te)
    add_fold(f, vi, pv, pt, t_fold, f' | train_min {train_min:.1f}')
    if f + 1 < K and not can_continue((time.time() - T0) / 60, fold_min, TIME_BUDGET_MIN):
        print(f'STOP EARLY after fold {f}', flush=True)
        break
save_run(True)'''
PAR = 'run_parallel(splits)\nsave_run(True)'
cell3 = f"""freq = np.bincount(train.y, minlength=3) / len(train)
class_w = torch.tensor((1 / freq) ** 0.5, dtype=torch.float)
class_w /= class_w.mean()
ta_params = inspect.signature(TrainingArguments).parameters

{FOLD_LIB}
{CORE}{PARALLEL_CODE.replace('@WORKER@', WORKER) if a.parallel_folds else ''}
oof, pte, mask = np.zeros((len(train), 3)), np.zeros((len(test), 3)), np.zeros(len(train), bool)
fold_f1, fold_min = [], []
splits = list(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED).split(train, train.y, group))
{PAR if a.parallel_folds else SEQ}"""

cells = [
("markdown", md),
("code", """import gc, glob, inspect, json, math, os, subprocess, sys, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np, pandas as pd, torch, transformers
from datasets import Dataset
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

T0 = time.time()
HF_NAME, MIRROR, K, EPOCHS, LR, HEAD, TAIL, TIME_BUDGET_MIN, SMOKE, GC, FP16, PARALLEL = @CFG@
SEED = 42
LABELS = ['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']

# Tìm thư mục model duy nhất có config.json và file trọng số, bỏ qua onnx và .cache.
def find_model(root='/kaggle/input'):
    found = {os.path.dirname(p) for p in glob.glob(f'{root}/**/config.json', recursive=True)
             if '/onnx' not in p and '/.cache' not in p}
    found = [d for d in found if os.path.exists(f'{d}/model.safetensors') or os.path.exists(f'{d}/pytorch_model.bin')]
    assert len(found) == 1, found
    return found[0]

MODEL = find_model()
DATA = glob.glob('/kaggle/input/**/train.csv', recursive=True)[0].rsplit('/', 1)[0]
print(MODEL, DATA, 'SMOKE' if SMOKE else 'FULL')
print('gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      '| torch', torch.__version__, '| transformers', transformers.__version__)""".replace('@CFG@', cfg)),
("code", """train = pd.read_csv(f'{DATA}/train.csv', keep_default_na=False)
test = pd.read_csv(f'{DATA}/test.csv', keep_default_na=False)
if SMOKE:
    train, test, K, EPOCHS = train.sample(800, random_state=SEED).reset_index(drop=True), test.head(300), 2, 1
train['y'] = train.label.map(LABELS.index)
group = train.text.str.lower().str.replace(r'\\W+', ' ', regex=True).str.strip().str[:60]
tok = AutoTokenizer.from_pretrained(MODEL)

# Chuyển danh sách text thành input_ids có token đặc biệt, giữ phần đầu và phần cuối khi quá dài.
def encode(texts):
    ids = tok(list(texts))['input_ids']
    return [x if len(x) <= HEAD + TAIL + 2 else x[:HEAD + 1] + x[-(TAIL + 1):] for x in ids]

# Gói input_ids (và nhãn nếu có) thành Dataset.
def to_ds(texts, y=None):
    d = {'input_ids': encode(texts)}
    if y is not None:
        d['labels'] = list(y)
    return Dataset.from_dict(d)

ds_all, ds_te = to_ds(train.text, train.y), to_ds(test.text)
print(len(train), len(test), 'groups', group.nunique())"""),
("code", cell3),
]
def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(True)}
    if t == "code": c.update(outputs=[], execution_count=None)
    return c
nb = {"cells": [cell(*c) for c in cells], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
if a.out.endswith('.ipynb'):
    nb_path = a.out
else:
    os.makedirs(a.out, exist_ok=True)
    nb_path = f'{a.out}/{code_file}'
    meta = dict(id=kid, title=title, code_file=code_file, language='python', kernel_type='notebook', is_private=True,
                enable_gpu=True, enable_internet=False, dataset_sources=[], kernel_sources=[],
                competition_sources=['rmit-hackathon-2026'], model_sources=[source])
    json.dump(meta, open(f'{a.out}/kernel-metadata.json', 'w'), indent=2)
json.dump(nb, open(nb_path, 'w'), ensure_ascii=False, indent=1)
