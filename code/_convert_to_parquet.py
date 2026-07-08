# xlsx -> parquet 변환 (1회성). Rust 기반 calamine 엔진으로 고속 처리.
# 대회 원본 데이터는 외부 공개 금지 -> 생성되는 parquet도 .gitignore 처리됨.
import glob, time, pandas as pd

t0 = time.time()
files = {f: ('발생' if '보상' not in f else '보상') for f in sorted(glob.glob('data/*.xlsx'))}
for f, kind in files.items():
    xls = pd.ExcelFile(f, engine='calamine')
    years = [s for s in xls.sheet_names if s.isdigit()]
    parts = []
    for y in years:
        df = pd.read_excel(xls, sheet_name=y, dtype=str)
        df['접수연도'] = int(y)
        parts.append(df)
        print(kind, y, df.shape, round(time.time() - t0, 1), 's', flush=True)
    full = pd.concat(parts, ignore_index=True)
    out = 'data/' + kind + '.parquet'
    full.to_parquet(out, index=False)
    print('SAVED', out, full.shape, round(time.time() - t0, 1), 's', flush=True)
print('ALL DONE', round(time.time() - t0, 1), 's', flush=True)
