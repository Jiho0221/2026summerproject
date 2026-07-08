# parquet 기반 전체 프로파일링 -> UTF-8 리포트로 저장 (콘솔 한글 깨짐 회피).
# 태스크 1~5의 실제 수치를 한 번에 산출. 노트북/리포트 서술의 근거로 사용.
import pandas as pd, numpy as np, io, re

SP = 'C:/Users/ekf44/AppData/Local/Temp/claude/C--Users-ekf44-Desktop-baf-summervac/7e8e3426-1950-42fa-8b5a-b44ee64a763c/scratchpad/'
o = io.open(SP + 'profile.txt', 'w', encoding='utf-8')
def w(*a): o.write(' '.join(str(x) for x in a) + '\n')

occ = pd.read_parquet('data/발생.parquet')
comp = pd.read_parquet('data/보상.parquet')

for name, df in [('발생', occ), ('보상', comp)]:
    w('#'*60); w('DATASET', name, 'shape', df.shape)
    w('-- dtypes/결측/유니크 --')
    for c in df.columns:
        w(f'  {c}: dtype={df[c].dtype}, null={df[c].isna().sum()} ({df[c].isna().mean()*100:.2f}%), nunique={df[c].nunique()}')
    dup = df.duplicated().sum()
    w('-- 완전중복행:', dup, f'({dup/len(df)*100:.2f}%)')

# ── 범주형 값 분포 ─────────────────────────────
CATS = ['지역','학교급','사고자구분','사고자학년','사고자성별','사고요일',
        '사고시간','사고장소','사고형태','사고부위','사고당시활동']
for name, df in [('발생', occ), ('보상', comp)]:
    w('#'*60); w('범주 분포 -', name)
    for c in CATS:
        if c not in df.columns: continue
        vc = df[c].value_counts(dropna=False)
        w(f'--- {c} (nunique={df[c].nunique()}) top20 ---')
        for k,v in vc.head(20).items():
            w(f'   {k}: {v} ({v/len(df)*100:.2f}%)')

# ── 발생: 시간 파생 ─────────────────────────────
w('#'*60); w('발생 시간축')
occ['사고연'] = occ['사고연월'].str.slice(0,4)
occ['사고월'] = occ['사고연월'].str.slice(5,7)
w('-- 사고연(발생연도) 분포 --')
for k,v in occ['사고연'].value_counts().sort_index().items():
    w(f'   {k}: {v}')
w('-- 접수연도 분포 --')
for k,v in occ['접수연도'].value_counts().sort_index().items():
    w(f'   {k}: {v}')
w('-- 사고월(전체) --')
for k,v in occ['사고월'].value_counts().sort_index().items():
    w(f'   {k}월: {v}')
# 발생연도 2021~2025만 필터한 월분포
recent = occ[occ['사고연'].isin(['2021','2022','2023','2024','2025'])]
w('-- 발생연도 2021~2025 한정 연x월 pivot --')
piv = recent.groupby(['사고연','사고월']).size().unstack(0)
w(piv.to_string())
w('-- 요일 분포(발생연 2021~2025) --')
for k,v in recent['사고요일'].value_counts().items():
    w(f'   {k}: {v}')

# 시각 파싱
def parse_hour(s):
    if not isinstance(s,str): return np.nan
    m = re.match(r'^\s*(\d{1,2})[:시]', s)
    if m:
        h=int(m.group(1))
        return h if 0<=h<=23 else np.nan
    return np.nan
occ['시'] = occ['사고발생시각'].map(parse_hour)
w('-- 사고발생시각 파싱: 결측/이상', occ['시'].isna().sum(), f'({occ["시"].isna().mean()*100:.2f}%)')
w('-- 시간대 분포(발생연 2021~2025) --')
rh = occ[occ['사고연'].isin(['2021','2022','2023','2024','2025'])]
for k,v in rh['시'].value_counts().sort_index().items():
    w(f'   {int(k):02d}시: {v}')

# ── 보상: 금액 ─────────────────────────────────
w('#'*60); w('보상 금액')
MONEY = ['요양급여','장해급여','간병급여','유족급여','장례비','위로금','보전비용']
for c in MONEY:
    comp[c] = pd.to_numeric(comp[c], errors='coerce').fillna(0)
comp['총보상'] = comp[MONEY].sum(axis=1)
w('-- 급여항목별 합계/지급건수(>0)/평균(>0) --')
for c in MONEY:
    nz = comp[comp[c]>0][c]
    w(f'   {c}: 합계={comp[c].sum():,.0f} | 건수={len(nz)} | 평균(지급건)={nz.mean() if len(nz) else 0:,.0f} | 최대={comp[c].max():,.0f}')
w('-- 총보상 통계 --')
w(comp['총보상'].describe().to_string())
for q in [0.5,0.9,0.95,0.99,0.999]:
    w(f'   분위 {q}: {comp["총보상"].quantile(q):,.0f}')
w('-- 총보상 0원 건수:', (comp['총보상']==0).sum(), f'({(comp["총보상"]==0).mean()*100:.2f}%)')
w('-- 장해/유족/간병 발생건수:', (comp['장해급여']>0).sum(), (comp['유족급여']>0).sum(), (comp['간병급여']>0).sum())

# 유형별 평균 보상 (심각도 대리지표)
for dim in ['사고형태','사고장소','사고당시활동','학교급','사고부위']:
    g = comp.groupby(dim)['총보상'].agg(['count','sum','mean']).sort_values('mean',ascending=False)
    g = g[g['count']>=200]  # 표본 충분한 것만
    w(f'--- {dim}별 평균 총보상 top10 (건수>=200) ---')
    w(g.head(10).to_string())
    w(f'--- {dim}별 총보상 합계 top10 ---')
    w(g.sort_values('sum',ascending=False).head(10).to_string())

o.close()
print('PROFILE DONE')
