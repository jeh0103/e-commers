# pages/02_Risky_List.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from urllib.parse import quote

st.set_page_config(page_title="📋 이탈 고객 리스트", layout="wide")

# ===== 화면 표시용 한글 라벨 =====
KOR_COL = {
    "CustomerID_clean": "고객ID",
    "GenderLabel": "성별",
    "Age": "나이",
    "CustomerLifetimeValue": "고객생애가치(CLV)",
    "TotalPurchases": "총 구매 횟수",
    "PurchaseFrequency": "구매 빈도(월 평균)",
    "CSFrequency": "상담 빈도(월 평균)",
    "AverageSatisfactionScore": "평균 만족도",
    "NegativeExperienceIndex": "부정 경험 지수",
    "EmailEngagementRate": "이메일 참여율",
    "TotalEngagementScore": "총 활동 점수",
    "ChurnRiskScore": "이탈 위험 점수",
    "RepeatAndPremiumFlag": "리피트/프리미엄 여부",
    # 내부적으로만 사용하는 모델 점수(화면 표에서는 숨김)
    "IF_AnomalyScore": "IF 이상치점수",
    "AE_ReconError": "AE 재구성오차",
}

def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: KOR_COL.get(c, c) for c in df.columns})

# ===== 성별 라벨 보장 =====
DEFAULT_CODE_TO_LABEL_KO = {1:"여성",3:"남성",5:"응답거부",4:"기타/미상",2:"남성",0:"여성"}

def _normalize_gender_text_to_label_ko(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "미상"
    s = str(x).strip().lower()
    if s in {"m","male","man","남","남성"}: return "남성"
    if s in {"f","female","woman","여","여성"}: return "여성"
    if s in {"prefer not to say","decline to state","no answer"}: return "응답거부"
    if s in {"non-binary","nonbinary","genderqueer","agender","nb","other","기타"}: return "기타"
    return "기타"

def ensure_gender_label(df_hybrid: pd.DataFrame,
                        original_csv_path: str = "ecommerce_customer_data.csv",
                        code_map_path: str = "gender_code_map.json") -> pd.DataFrame:
    df = df_hybrid.copy()
    if os.path.exists(original_csv_path):
        try:
            raw = pd.read_csv(original_csv_path, usecols=["CustomerID","Gender"])
            raw["GenderLabel_from_raw"] = raw["Gender"].map(_normalize_gender_text_to_label_ko)
            df = df.merge(raw[["CustomerID","GenderLabel_from_raw"]], on="CustomerID", how="left")
        except Exception:
            df["GenderLabel_from_raw"] = np.nan
    else:
        df["GenderLabel_from_raw"] = np.nan
    code_map = DEFAULT_CODE_TO_LABEL_KO.copy()
    if os.path.exists(code_map_path):
        try:
            with open(code_map_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                code_map.update({int(k): v for k, v in loaded.items()})
        except Exception:
            pass
    label_from_code = df["Gender"].map(code_map) if "Gender" in df.columns else pd.Series(index=df.index, dtype="object")
    df["GenderLabel"] = df["GenderLabel_from_raw"].fillna(label_from_code)
    df.drop(columns=["GenderLabel_from_raw"], inplace=True)
    df["GenderLabel"] = df["GenderLabel"].fillna("미상")
    return df

# ===== 데이터 로딩 =====
@st.cache_data(show_spinner=False)
def load_main():
    df = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")

    # CustomerID_clean 보장
    def _clean_id(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        return np.nan if (s == "" or s.lower() in {"nan", "none", "nat", "null"}) else s

    if "CustomerID" in df.columns:
        df["CustomerID_clean"] = df["CustomerID"].map(_clean_id)
    else:
        df["CustomerID_clean"] = np.nan

    
    if df["CustomerID_clean"].isna().all() or df["CustomerID_clean"].isna().any():
        generated = pd.Series(np.arange(1, len(df) + 1), index=df.index).map(lambda i: f"CUST{i:05d}")
        df["CustomerID_clean"] = df["CustomerID_clean"].fillna(generated)
        if "CustomerID" not in df.columns:
            df["CustomerID"] = df["CustomerID_clean"]

    df = ensure_gender_label(df)
    return df
df = load_main()

# ===== 전역 필터/임계값 세션 값 재사용 =====
sel_age = st.session_state.get("sel_age")
sel_gender_labels = st.session_state.get("sel_gender_labels", [])
premium_opt = st.session_state.get("premium_opt", "전체")
use_dynamic = bool(st.session_state.get("use_dynamic", False))
if_thr = st.session_state.get("if_thr")
ae_thr = st.session_state.get("ae_thr")

# 필터 적용
filtered = df.copy()
if sel_age:
    filtered = filtered[(filtered["Age"] >= sel_age[0]) & (filtered["Age"] <= sel_age[1])]
if sel_gender_labels:
    filtered = filtered[filtered["GenderLabel"].isin(sel_gender_labels)]
if "RepeatAndPremiumFlag" in filtered.columns and premium_opt != "전체":
    filtered = filtered[filtered["RepeatAndPremiumFlag"] == (1 if str(premium_opt).startswith("예") else 0)]

# ===== 파라미터: src (if|ae|both) =====
src = st.query_params.get("src", "both") if hasattr(st, "query_params") \
      else st.experimental_get_query_params().get("src", ["both"])[0]
src = (src if isinstance(src, str) else src[0]).lower()

# ===== 상단 네비 =====
try:
    st.page_link("app_enhanced.py", label="⬅️ 대시보드로", icon="🏠")
except Exception:
    st.markdown("[🏠 대시보드로](/)")

TITLE = {
    "if":   "이상행동 기반 이탈 의심 고객",
    "ae":   "패턴 변화 기반 이탈 의심 고객",
    "both": "공통 이탈 고객(고신뢰군)"
}
st.title(f"🗂️ {TITLE.get(src, '고객 리스트')}")

# ===== 판단 기준/기본 설정 =====
if src == "if":
    flag_col = "IF_ChurnFlag_dyn" if (use_dynamic and "IF_ChurnFlag_dyn" in filtered.columns) else "IF_ChurnFlag"
    sort_metric = "IF_AnomalyScore" if "IF_AnomalyScore" in filtered.columns else "ChurnRiskScore"
    subset = filtered[filtered.get(flag_col, 0) == 1] if flag_col in filtered.columns else filtered.copy()
    thr_value = float(if_thr) if (use_dynamic and if_thr is not None) else (
        float(filtered["IF_AnomalyScore"].quantile(0.95)) if "IF_AnomalyScore" in filtered.columns else None)
    st.markdown(
        "**판단 기준 안내**\n\n"
        "- Isolation Forest는 격리 깊이로 이상치 점수를 계산하며, 점수가 클수록 이탈 신호로 간주합니다.\n"
        "- 아래 목록은 해당 기준을 충족한 고객을 **위험도 순**으로 정렬해 보여줍니다."
    )
elif src == "ae":
    flag_col = "AE_ChurnFlag_dyn" if (use_dynamic and "AE_ChurnFlag_dyn" in filtered.columns) else "AE_ChurnFlag"
    sort_metric = "AE_ReconError" if "AE_ReconError" in filtered.columns else "ChurnRiskScore"
    subset = filtered[filtered.get(flag_col, 0) == 1] if flag_col in filtered.columns else filtered.copy()
    thr_value = float(ae_thr) if (use_dynamic and ae_thr is not None) else (
        float(filtered["AE_ReconError"].quantile(0.95)) if "AE_ReconError" in filtered.columns else None)
    st.markdown(
        "**판단 기준 안내**\n\n"
        "- Autoencoder는 정상 패턴 대비 재구성 오차가 큰 샘플을 이탈 신호로 간주합니다.\n"
        "- 아래 목록은 해당 기준을 충족한 고객을 **위험도 순**으로 정렬해 보여줍니다."
    )
else:
    flag_col = "Both_ChurnFlag_dyn" if (use_dynamic and "Both_ChurnFlag_dyn" in filtered.columns) else "Both_ChurnFlag"
    sort_metric = "ChurnRiskScore"
    subset = filtered[filtered.get(flag_col, 0) == 1] if flag_col in filtered.columns else filtered.copy()
    thr_value = None
    st.markdown(
        "**판단 기준 안내**\n\n"
        "- **두 모델 모두 이탈**로 판단된 고객을 고신뢰군으로 정의합니다.\n"
        "- 아래 목록은 고신뢰군 중 **이탈위험점수**가 높은 순으로 정렬합니다."
    )

# ===== 스냅샷 패널 =====
colA, colB, colC = st.columns(3)
target_n = int(len(subset))
total_n = int(len(filtered)) if len(filtered) else 1
colA.metric("대상 고객 수", f"{target_n:,}", f"{(target_n/total_n*100):.2f}%")
if thr_value is not None and np.isfinite(thr_value):
    colB.metric("사용 임계값", f"{thr_value:.4f}")
else:
    colB.metric("사용 임계값", "—")
if sort_metric in filtered.columns:
    s_all = pd.to_numeric(filtered[sort_metric], errors="coerce")
    s_sub = pd.to_numeric(subset[sort_metric], errors="coerce")
    m_all = float(s_all.mean()) if s_all.notna().any() else 0.0
    m_sub = float(s_sub.mean()) if s_sub.notna().any() else 0.0
    delta_pct = ((m_sub - m_all)/m_all*100.0) if m_all > 0 else 0.0
    colC.metric(
        f"{KOR_COL.get(sort_metric, sort_metric)} 평균",
        f"{m_sub:.4f}",
        f"{delta_pct:+.1f}% vs 전체"
    )

# ===== 한글 폰트 자동 설정 (그래프용) =====
def _set_korean_font_if_available():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        candidates = [
            "Apple SD Gothic Neo", "Malgun Gothic",
            "NanumGothic", "Nanum Gothic", "Noto Sans CJK KR"
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# ===== (선택) 그래프 보기 + 자동 해석 =====
show_plot = st.toggle("그래프 보기(선택)", value=False)
if show_plot and (sort_metric in filtered.columns):
    try:
        import matplotlib.pyplot as plt
        _set_korean_font_if_available()

        vals = pd.to_numeric(filtered[sort_metric], errors="coerce").dropna()
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=120)
            ax.hist(vals, bins=30)
            title_key = KOR_COL.get(sort_metric, sort_metric)
            if thr_value is not None and np.isfinite(thr_value):
                ax.axvline(thr_value, linestyle="--")
                ax.set_title(f"{title_key} 분포 (점선=임계)", fontsize=14, pad=8)
            else:
                ax.set_title(f"{title_key} 분포", fontsize=14, pad=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            q50 = float(vals.quantile(0.50))
            q90 = float(vals.quantile(0.90))
            q95 = float(vals.quantile(0.95))
            mean = float(vals.mean())
            std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            skew = float(vals.skew()) if len(vals) > 2 else 0.0
            skew_txt = (
                "우측 꼬리(큰 값에 소수 집중)" if skew > 0.5
                else ("좌측 꼬리(작은 값에 소수 집중)" if skew < -0.5 else "대칭에 가까움")
            )

            if thr_value is not None and np.isfinite(thr_value):
                above = int((vals >= thr_value).sum())
                pct_above = 100.0 * above / len(vals)
                thr_pct = 100.0 * (vals <= thr_value).mean()
                st.markdown(
                    f"""
**그래프 해석**
- 분포 형태: **{skew_txt}** *(skew={skew:.2f})*
- 중앙값/상위 분위(90/95): **{q50:.4f} / {q90:.4f} / {q95:.4f}**
- 임계값 위치: 데이터의 **약 {thr_pct:.1f}퍼센타일**
- 임계 이상 고객 비중: **{above:,}명 ({pct_above:.2f}%)**
                    """.strip()
                )
            else:
                st.markdown(
                    f"""
**그래프 해석**
- 분포 형태: **{skew_txt}** *(skew={skew:.2f})*
- 중앙값/상위 분위(90/95): **{q50:.4f} / {q90:.4f} / {q95:.4f}**
- 현재 섹션은 임계값 없이 **이탈위험점수 상위** 기준으로 목록이 정렬됩니다.
                    """.strip()
                )

            if src == "if":
                st.caption(
                    "ℹ️ IF 점수는 격리 깊이에 기반합니다. "
                    "임계값을 낮추면 탐지 폭이 넓어지고(재현율↑), 높이면 엄격해집니다(정밀도↑)."
                )
            elif src == "ae":
                st.caption(
                    "ℹ️ AE 오차는 정상 패턴에서 벗어난 정도입니다. "
                    "임계값을 낮추면 더 많은 이상 신호를 포착합니다."
                )
            else:
                st.caption(
                    "ℹ️ 고신뢰군은 IF와 AE 모두 임계 이상인 고객입니다. "
                    "상단 표의 ‘리스크요인’ 태그로 관리 우선순위를 확인하세요."
                )
    except Exception:
        pass

st.markdown("---")

# ===== 위험도 순 리스트 (리스크 요인 태그 + 우선 연락도 지표)
# 정렬 및 순위점수 생성
if sort_metric in subset.columns:
    subset = subset.sort_values(sort_metric, ascending=False)
    subset["__rank_score__"] = subset[sort_metric]
elif "ChurnRiskScore" in subset.columns:
    subset = subset.sort_values("ChurnRiskScore", ascending=False)
    subset["__rank_score__"] = subset["ChurnRiskScore"]
else:
    subset["__rank_score__"] = 0.0

# 고객ID 결측 제거
if "CustomerID_clean" in subset.columns:
    subset = subset[subset["CustomerID_clean"].notna()]
elif "CustomerID" in subset.columns:
    subset = subset[subset["CustomerID"].notna()]

top_k = st.slider("표시 건수", min_value=10, max_value=500, value=100, step=10)

# 리스크 태그 기준(분위) 계산 — 전체(필터적용 후) 분포 기준
def qdict(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return None
    return {
        "p10": float(s.quantile(0.10)), "p20": float(s.quantile(0.20)),
        "p80": float(s.quantile(0.80)), "p90": float(s.quantile(0.90))
    }

q = {}
for c in [
    "NegativeExperienceIndex","AverageSatisfactionScore","EmailEngagementRate",
    "CSFrequency","TotalEngagementScore","AvgPurchaseInterval","PurchaseFrequency"
]:
    if c in filtered.columns:
        q[c] = qdict(filtered[c])

def make_risk_tags(row) -> tuple[str, str]:
    tags_html, tags_text = [], []
    def add(label, color):
        tags_html.append(f"<span class='tag tag-{color}'>{label}</span>")
        tags_text.append(label)

    ne = row.get("NegativeExperienceIndex")
    if ne is not None and "NegativeExperienceIndex" in q and q["NegativeExperienceIndex"]:
        if pd.notna(ne) and ne >= q["NegativeExperienceIndex"]["p80"]:
            add("부정경험↑", "red")

    sat = row.get("AverageSatisfactionScore")
    if sat is not None and "AverageSatisfactionScore" in q and q["AverageSatisfactionScore"]:
        if pd.notna(sat) and sat <= q["AverageSatisfactionScore"]["p20"]:
            add("만족도↓", "amber")

    em = row.get("EmailEngagementRate")
    if em is not None and "EmailEngagementRate" in q and q["EmailEngagementRate"]:
        if pd.notna(em) and em <= q["EmailEngagementRate"]["p20"]:
            add("이메일참여↓", "amber")

    cs = row.get("CSFrequency")
    if cs is not None and "CSFrequency" in q and q["CSFrequency"]:
        if pd.notna(cs) and cs >= q["CSFrequency"]["p80"]:
            add("상담빈도↑", "amber")

    te = row.get("TotalEngagementScore")
    if te is not None and "TotalEngagementScore" in q and q["TotalEngagementScore"]:
        if pd.notna(te) and te <= q["TotalEngagementScore"]["p20"]:
            add("참여점수↓", "gray")

    ap = row.get("AvgPurchaseInterval")
    if ap is not None and "AvgPurchaseInterval" in q and q["AvgPurchaseInterval"]:
        if pd.notna(ap) and ap >= q["AvgPurchaseInterval"]["p80"]:
            add("구매간격↑", "gray")

    pf = row.get("PurchaseFrequency")
    if pf is not None and "PurchaseFrequency" in q and q["PurchaseFrequency"]:
        if pd.notna(pf) and pf <= q["PurchaseFrequency"]["p20"]:
            add("구매빈도↓", "gray")

    return " ".join(tags_html), ", ".join(tags_text)

top_sub = subset.head(top_k).copy()
html_tags, text_tags = [], []
for _, r in top_sub.iterrows():
    h, t = make_risk_tags(r)
    html_tags.append(h)
    text_tags.append(t)
top_sub["__tags_html__"] = html_tags
top_sub["__tags_text__"] = text_tags

# ===== 우선 연락도(0-100) 계산 (5~95 분위 기준 정규화)
def _priority_index_from_quantiles(ref_series: pd.Series, values: pd.Series,
                                   q_low=0.05, q_high=0.95) -> pd.Series:
    ref = pd.to_numeric(ref_series, errors="coerce")
    val = pd.to_numeric(values, errors="coerce")
    if ref.notna().any():
        lo = float(ref.quantile(q_low)); hi = float(ref.quantile(q_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(ref.min()), float(ref.max())
    else:
        lo, hi = 0.0, 1.0
    rng = hi - lo if (hi > lo) else 1.0
    idx = ((val - lo) / rng).clip(0, 1) * 100.0
    return idx.round(0).fillna(0)

if sort_metric in filtered.columns:
    top_sub["__priority_idx__"] = _priority_index_from_quantiles(
        filtered[sort_metric], top_sub[sort_metric]
    )
else:
    top_sub["__priority_idx__"] = 0

# 우선 연락도 HTML(막대+배지) 생성
def _priority_tier(idx: float):
    if idx >= 90: return "최우선", "rb-red"
    if idx >= 70: return "높음", "rb-orange"
    if idx >= 40: return "보통", "rb-amber"
    return "후순위", "rb-gray"

def _mk_priority_html(idx: float, raw: float, thr: float | None):
    """우선 연락도 표시용 HTML – 배지만 표시(점수 텍스트 없음)."""
    label, css = _priority_tier(float(idx))
    tip = f"우선 연락 점수 {int(idx)}/100"
    if pd.notna(raw):
        tip += f" | 모델 원점수 {float(raw):.4f}"
    if thr is not None and np.isfinite(thr):
        tip += f" | 임계 {float(thr):.4f}"

    # ✅ 배지 하나만 렌더링 (최우선/높음/보통/후순위)
    return f"<span class='rbadge {css}' title='{tip}'>{label}</span>"

top_sub["__priority_html__"] = [
    _mk_priority_html(
        idx,
        raw=top_sub.iloc[i][sort_metric] if sort_metric in top_sub.columns else np.nan,
        thr=thr_value
    )
    for i, idx in enumerate(top_sub["__priority_idx__"])
]

# ===== 표 구성 (관리자 친화)
desired = [
    "CustomerID_clean",
    "GenderLabel",
    "Age",
    "RepeatAndPremiumFlag",
    "CustomerLifetimeValue",
    "TotalPurchases",
    "PurchaseFrequency",
    "CSFrequency",
    "AverageSatisfactionScore",
    "NegativeExperienceIndex",
    "EmailEngagementRate",
    "TotalEngagementScore",
    "ChurnRiskScore",
    "__priority_idx__",
    "__priority_html__",
]
cols_to_show = [c for c in desired if c in top_sub.columns]
view_df = top_sub[cols_to_show].copy()

if view_df.empty:
    st.info("현재 조건에서 표시할 고객이 없습니다.")
    st.stop()

# 순번 + 상세 링크 + 리스크 요인 + 우선 연락도(HTML)
view_df.insert(0, "", np.arange(1, len(view_df) + 1))
view_df["고객ID"] = top_sub["CustomerID_clean"].apply(
    lambda cid: f"<a href='/Customer_Detail?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
)
view_df["리스크요인"] = top_sub["__tags_html__"]
view_df["우선 연락도"] = top_sub["__priority_html__"]

# 불필요한 내부 컬럼 제거 및 라벨링
view_df.drop(columns=["CustomerID_clean","__priority_html__","__priority_idx__"], inplace=True, errors="ignore")
view_df = rename_for_display(view_df)

# 표 표시 순서: 순위 → 고객ID → 우선 연락도 → 리스크요인 → 나머지
display_cols = ["", "고객ID", "우선 연락도", "리스크요인"] + [
    c for c in view_df.columns if c not in ("","고객ID","우선 연락도","리스크요인")
]

# 숫자 포맷
age_label = KOR_COL.get("Age", "Age")
clv_label = KOR_COL.get("CustomerLifetimeValue", "CustomerLifetimeValue")
tp_label  = KOR_COL.get("TotalPurchases", "TotalPurchases")

fmt_map = {}
for c in display_cols:
    if c in ("","고객ID","성별","우선 연락도","리스크요인"):
        continue
    if c in (age_label, tp_label):
        fmt_map[c] = "{:.0f}"
    elif c == clv_label:
        fmt_map[c] = "{:,.0f}"
    else:
        fmt_map[c] = "{:.2f}"

styler = (
    view_df[display_cols]
    .style
    .format(fmt_map)
    .hide(axis="index")
    .set_table_attributes('class="dataframe"')
)

table_html = styler.to_html(escape=False)

st.markdown(
    """
    <style>
    /* 가로 스크롤 컨테이너 */
    .risky-scroll {
      width: 100%;
      overflow-x: auto;       /* 🔥 여기서 가로 스크롤 강제 */
    }

    .risky-scroll table {
      border-collapse: collapse;
      width: auto !important;
      min-width: 1500px;      /* 화면보다 넓게 만들어야 스크롤이 생김. 필요하면 1800 등으로 조정 */
      max-width: none !important;
      table-layout: auto;
    }

    .risky-scroll th,
    .risky-scroll td {
      padding: 10px 12px !important;
      line-height: 1.45;
      vertical-align: middle;
      white-space: nowrap;    /* 줄바꿈 대신 가로로 쭉 펼침 */
    }

    /* 리스크 요인 태그 */
    .tag {
      display: inline-block;
      padding: 2px 6px;
      margin-right: 4px;
      margin-bottom: 2px;
      border-radius: 6px;
      font-size: 12px;
    }
    .tag-red   { background: rgba(255, 59, 48, 0.18); border: 1px solid rgba(255, 59, 48, 0.35); }
    .tag-amber { background: rgba(255,149,  0, 0.18); border: 1px solid rgba(255,149,  0, 0.35); }
    .tag-gray  { background: rgba(128,128,128,0.18); border: 1px solid rgba(128,128,128,0.35); }

    /* 우선 연락도 막대 + 배지 */
    .rwrap { display:flex; align-items:center; gap:8px; }
    .rbar  { flex:1; height:10px; background:rgba(0,0,0,0.06); border-radius:999px; overflow:hidden; }
    .rbar .fill { height:100%; }
    .fill.rb-red    { background: rgba(255, 59, 48, 0.60); }
    .fill.rb-orange { background: rgba(255,149,  0, 0.60); }
    .fill.rb-amber  { background: rgba(255,204,  0, 0.55); }
    .fill.rb-gray   { background: rgba(128,128,128,0.45); }

    .rbadge {
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1;
      border:1px solid transparent;
    }
    .rbadge.rb-red    { background: rgba(255, 59, 48, 0.18); border-color: rgba(255, 59, 48, 0.35); }
    .rbadge.rb-orange { background: rgba(255,149,  0, 0.18); border-color: rgba(255,149,  0, 0.35); }
    .rbadge.rb-amber  { background: rgba(255,204,  0, 0.18); border-color: rgba(255,204,  0, 0.35); }
    .rbadge.rb-gray   { background: rgba(128,128,128,0.18); border-color: rgba(128,128,128,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='risky-scroll'>{table_html}</div>",
    unsafe_allow_html=True,
)

# ===== CSV 다운로드 (태그=텍스트, 우선 연락도/원점수 포함) =====
export_df = view_df.copy()
export_df.rename(columns={"": "순위"}, inplace=True)
export_df.insert(1, "CustomerID", export_df["고객ID"].str.extract(r'>(.*?)<')[0])

export_df.drop(columns=["우선 연락도"], inplace=True)
export_df.insert(2, "우선연락도(0-100)", top_sub["__priority_idx__"].astype(int).values)

raw_label = {
    "if":   f"원점수({KOR_COL.get('IF_AnomalyScore','IF_AnomalyScore')})",
    "ae":   f"원점수({KOR_COL.get('AE_ReconError','AE_ReconError')})",
    "both": f"원점수({KOR_COL.get('ChurnRiskScore','ChurnRiskScore')})",
}.get(src, "원점수")
raw_series = (
    pd.to_numeric(top_sub[sort_metric], errors="coerce").round(6)
    if sort_metric in top_sub.columns else pd.Series([np.nan]*len(top_sub))
)
export_df.insert(3, raw_label, raw_series.values)
export_df["리스크요인"] = top_sub["__tags_text__"].values

csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ CSV 내려받기",
    data=csv_bytes,
    file_name=f"{src}_risky_customers.csv",
    mime="text/csv"
)