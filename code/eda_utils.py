"""EDA 공통 유틸 (노트북에서 import).

- 대회 원본 데이터는 외부 공개 금지. 이 모듈/노트북에 원본 값을 하드코딩하지 않는다.
- parquet(_convert_to_parquet.py 산출)을 읽어 분석에 사용.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── 한글 폰트 (Windows: 맑은 고딕) ──────────────────────────
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['savefig.bbox'] = 'tight'

MONEY = ['요양급여', '장해급여', '간병급여', '유족급여', '장례비', '위로금', '보전비용']
RECENT_YEARS = ['2021', '2022', '2023', '2024', '2025']

# 방학 기간(대략): 겨울 12~2월, 여름 7~8월 → '방학' vs '학기' 대비용
VACATION_MONTHS = {'01', '02', '07', '08', '12'}


def load_occ(path='../data/발생.parquet'):
    """발생 데이터 로드 + 시간 파생 컬럼 추가."""
    df = pd.read_parquet(path)
    df['사고연'] = df['사고연월'].str.slice(0, 4)
    df['사고월'] = df['사고연월'].str.slice(5, 7)
    df['시'] = df['사고발생시각'].map(_parse_hour)
    df['방학여부'] = np.where(df['사고월'].isin(VACATION_MONTHS), '방학(추정)', '학기중')
    return df


def load_comp(path='../data/보상.parquet'):
    """보상 데이터 로드 + 금액 숫자화 + 총보상 컬럼."""
    df = pd.read_parquet(path)
    for c in MONEY:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['총보상'] = df[MONEY].sum(axis=1)
    return df


def _parse_hour(s):
    """'12:40', '12:8', '9:45', '13시' 등에서 시(0~23) 추출."""
    if not isinstance(s, str):
        return np.nan
    m = re.match(r'^\s*(\d{1,2})[:시]', s)
    if m:
        h = int(m.group(1))
        return h if 0 <= h <= 23 else np.nan
    return np.nan


def krw(x):
    """원 단위 → 읽기 쉬운 문자열."""
    return f'{x:,.0f}원'
