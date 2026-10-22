import argparse, os, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
ap.add_argument('--n', type=int, help='chỉ dùng n dòng đầu (chạy thử nhanh)')
a = ap.parse_args()

# Tìm thư mục chứa train.csv dưới thư mục dữ liệu đã giải nén.
def find_train(root):
    for d, _, files in os.walk(root):
        if 'train.csv' in files:
            return os.path.join(d, 'train.csv')
    raise FileNotFoundError(f'không thấy train.csv dưới {root}')

LABELS = ['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']
train = pd.read_csv(find_train(a.data), keep_default_na=False)
if a.n:
    train = train.sample(a.n, random_state=42).reset_index(drop=True)
y = train.label.map(LABELS.index)
group = train.text.str.lower().str.replace(r'\W+', ' ', regex=True).str.strip().str[:60]
print(len(train), 'rows', group.nunique(), 'groups', flush=True)

model = make_pipeline(
    FeatureUnion([('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                  ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))]),
    LogisticRegression(C=10, class_weight='balanced', max_iter=300))
for name, cv, kw in [('StratifiedGroupKFold', StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), dict(groups=group)),
                     ('StratifiedKFold', StratifiedKFold(n_splits=3, shuffle=True, random_state=42), {})]:
    t = time.time()
    pred = cross_val_predict(model, train.text, y, cv=cv, **kw)
    print(f'{name} OOF macro F1 {f1_score(y, pred, average="macro"):.4f} | {(time.time() - t) / 60:.1f} min', flush=True)
