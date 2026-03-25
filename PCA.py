import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# --- 용어 설명 딕셔너리 (툴팁 내용) ---
TOOLTIPS = {
    "LSL": "하한규격 (Lower Specification Limit)<br>제품이 가져야 할 최소 허용치입니다.",
    "Target": "목표치 (Target)<br>공정이 달성하고자 하는 이상적인 중심값입니다.",
    "USL": "상한규격 (Upper Specification Limit)<br>제품이 가져야 할 최대 허용치입니다.",
    "Sample N": "시료 수 (Sample Size)<br>분석에 사용된 데이터의 총 개수입니다.",
    "Mean": "평균 (Mean)<br>데이터들의 중심 위치(산술 평균)입니다.",
    "StDev (Within)": "군내 표준편차 (단기)<br>부분군 내의 변동이나 이동범위(MR)를 통해 예측한 표준편차입니다.<br>Cpk 계산에 사용됩니다.",
    "StDev (Overall)": "전체 표준편차 (장기)<br>모든 데이터의 단순 표본 표준편차입니다.<br>Ppk 계산에 사용됩니다.",
    "Cp": "잠재적 공정능력 (단기)<br>치우침을 고려하지 않은 단기 공정 능력입니다.",
    "Cpk": "실제 공정능력 (단기)<br>치우침을 반영한 단기 공정 능력입니다. (미니탭 Cpk와 동일)",
    "Pp": "잠재적 공정성능 (장기)<br>치우침을 고려하지 않은 장기 공정 능력입니다.",
    "Ppk": "실제 공정성능 (장기)<br>치우침을 반영한 장기 공정 능력입니다. (미니탭 Ppk와 동일)",
    "Sigma Level": "시그마 수준 (Sigma Level)<br>공정의 불량률을 나타내는 지표로, 높을수록 불량이 적습니다.<br>(3 × Cpk)",
    "Subgroup Size": "부분군 크기 (Subgroup Size)<br>한 번 샘플링할 때 묶는 데이터의 개수입니다.",
    "UCL": "관리상한 (Upper Control Limit)<br>공정의 우연 원인에 의한 자연스러운 변동의 상한선입니다.",
    "CL": "중심선 (Center Line)<br>공정 데이터의 평균적인 수준을 나타내는 기준선입니다.",
    "LCL": "관리하한 (Lower Control Limit)<br>공정의 우연 원인에 의한 자연스러운 변동의 하한선입니다.",
    "AD Stat": "Anderson-Darling 통계량<br>데이터가 정규분포를 따르는지 검정하는 수치입니다.<br>값이 작을수록 정규분포에 가깝습니다.",
    "P-Value": "유의확률 (P-Value)<br>정규성 검정의 판단 기준입니다.<br>0.05 이상이면 정규분포를 따른다고 판단합니다."
}

# --- 미니탭 표준편차 추정을 위한 d2 상수표 ---
D2_TABLE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

def add_interactive_summary_box(fig, lines, x_pos=1.02, y_center=0.5, fig_height=650):
    PX_LINE_HEIGHT = 28     
    PX_SECTION_GAP = 5      
    PX_PADDING = 15         

    line_height_rel = PX_LINE_HEIGHT / fig_height
    section_gap_rel = PX_SECTION_GAP / fig_height
    padding_rel = PX_PADDING / fig_height

    valid_lines = [l for l in lines if l.get('label', '').strip() != ""]

    total_content_height = 0
    for i, item in enumerate(valid_lines):
        total_content_height += line_height_rel
        if item.get('is_header') and i > 0:
            total_content_height += section_gap_rel
    
    total_box_height = total_content_height + (padding_rel * 2)

    box_y_top = y_center + (total_box_height / 2)
    box_y_bottom = y_center - (total_box_height / 2)
    current_y = box_y_top - padding_rel

    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=x_pos, 
        x1=x_pos + 0.145, 
        y0=box_y_bottom, 
        y1=box_y_top,
        fillcolor="white", 
        line=dict(color="#D5D8DC", width=1)
    )

    for i, item in enumerate(valid_lines):
        label = item.get('label', '')
        val = item.get('value', '')
        is_header = item.get('is_header', False)

        if is_header and i > 0:
            current_y -= section_gap_rel

        if is_header:
            text_str = f"<b>{label}</b>"
            font_size = 13
            hover_text = None 
        else:
            text_str = f"{label}: {val}"
            font_size = 12
            key_full = label
            key_short = label.split(" (")[0]
            hover_text = TOOLTIPS.get(key_full, TOOLTIPS.get(key_short, text_str))

        fig.add_annotation(
            xref="paper", yref="paper",
            x=x_pos + 0.005,
            y=current_y,
            text=text_str,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=font_size, color="black"),
            hovertext=hover_text,
            bgcolor="rgba(0,0,0,0)"
        )
        current_y -= line_height_rel

# 1. 페이지 설정
st.set_page_config(page_title="Process Capability Analysis-HJ", layout="wide")

if 'current_col' not in st.session_state:
    st.session_state.current_col = None
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

st.markdown("""
    <style>
    div[data-baseweb="popover"] { min-width: 500px !important; max-width: 800px !important; }
    </style>
    """, unsafe_allow_html=True)
st.title("📊 Process Capability Analysis v0.3 (Minitab Std.)")

# 3. 데이터 가이드 섹션
with st.expander("ℹ️ 데이터 입력 형식 가이드 & 예시 파일 다운로드 (Click)", expanded=False):
    st.markdown("""
    ### 📂 데이터 준비 방법
    1. **첫 번째 행(Header):** 데이터의 이름(예: `Length`, `Weight`)을 적어주세요.
    2. **두 번째 행부터:** 실제 측정값(숫자)만 입력해주세요.
    """)
    example_df = pd.DataFrame({"Length (예시)": [0.402, 0.405, 0.398, 0.410, 0.401, "...", 0.403]})
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("#### 👀 데이터 미리보기")
        st.dataframe(example_df, hide_index=True, use_container_width=True)
    with c2:
        st.write("#### 💾 샘플 파일 다운로드")
        sample_csv_df = pd.DataFrame({"Measurement_Data": [0.426, 0.452, 0.413, 0.426, 0.413, 0.387, 0.452, 0.452, 0.401] * 10})
        csv_sample = sample_csv_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 예시 CSV 파일 다운로드", data=csv_sample, file_name="Sample_Data.csv", mime="text/csv", type="primary")

# 2. 사이드바 설정
with st.sidebar:
    st.header("⚙️ 분석 설정")
    input_method = st.radio("데이터 입력 방식", ["파일 업로드", "데이터 붙여넣기"])
    
    st.write("---")
    st.subheader("📏 규격치 (Specs)")
    st.caption("※ 하한이나 상한이 없는 경우, 숫자를 완전히 지워 빈칸으로 두세요.")
    
    # 빈칸 입력을 허용하기 위해 텍스트 인풋으로 변경
    lsl_str = st.text_input("하한규격 (LSL)", value="25.0")
    target_str = st.text_input("목표치 (Target)", value="")
    usl_str = st.text_input("상한규격 (USL)", value="")
    
    def parse_spec(val_str):
        if not val_str.strip(): return None
        try: return float(val_str.strip())
        except ValueError:
            st.sidebar.error("⚠️ 숫자로 변환할 수 없는 값이 포함되어 빈칸으로 처리됩니다.")
            return None

    lsl = parse_spec(lsl_str)
    target = parse_spec(target_str)
    usl = parse_spec(usl_str)
    
    st.write("---")
    st.subheader("🌐 그래프 컨트롤")
    
    st.write("**[X축 설정]**")
    x_axis_mode = st.radio("X축 범위 모드", ["자동 (Auto)", "수동 (Manual)"])
    if x_axis_mode == "수동 (Manual)":
        x_min_default = lsl - 5.0 if lsl is not None else 0.0
        x_max_default = usl + 5.0 if usl is not None else 50.0
        x_min_val = st.number_input("X축 최소값", value=st.session_state.get('auto_x_min', x_min_default), format="%.5f")
        x_max_val = st.number_input("X축 최대값", value=st.session_state.get('auto_x_max', x_max_default), format="%.5f")
        x_step = st.number_input("X축 눈금 단위 (Bin Size)", value=st.session_state.get('auto_x_step', 1.0), format="%.5f", min_value=0.00001)
    
    x_axis_title = st.text_input("X축 제목", value="Measurement Value")

    st.write("**[Y축 설정 (Histogram)]**")
    y_axis_mode = st.radio("Y축 범위 모드", ["자동 (Auto)", "수동 (Manual)"])
    if y_axis_mode == "수동 (Manual)":
        y_min_val = st.number_input("Y축 최소값", value=st.session_state.get('auto_y_min', 0.0), format="%.1f")
        y_max_val = st.number_input("Y축 최대값", value=st.session_state.get('auto_y_max', 50.0), format="%.1f")
        y_step = st.number_input("Y축 눈금 단위", value=st.session_state.get('auto_y_step', 5.0), format="%.1f", min_value=0.1)
        
    y_axis_title = st.text_input("Y축 제목", value="Frequency")    
    
    st.write("---")
    subgroup_size = st.number_input("관리도 및 군내변동 시료군(n) 크기", value=1, min_value=1, help="1로 설정 시 이동범위(Moving Range)로 Cpk를 계산합니다.")

# 3. 메인 화면 - 데이터 로드 로직
data = pd.Series(dtype=float)
column_name = ""

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("엑셀/CSV 파일을 업로드하세요 (첫 줄은 제목)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        column_name = st.selectbox("🎯 분석할 열(Column)을 선택하세요:", df.columns)
        data = df[column_name].dropna()
else:
    raw_data = st.text_area("데이터를 붙여넣으세요 (첫 줄은 제목)", height=150, value="")
    if raw_data:
        lines = raw_data.strip().split('\n')
        column_name = lines[0]
        try:
            data = pd.Series([float(x.strip()) for x in lines[1:] if x.strip()])
        except:
            st.error("데이터에 숫자가 아닌 값이 포함되어 있습니다.")

st.write("")
run_analysis = st.button("🚀 Process Capability Analysis Start", use_container_width=True, type="primary")

# 4. 분석 결과 및 시각화 로직
if not data.empty:
    if st.session_state.current_col != column_name:
        st.session_state.current_col = column_name
        st.session_state.analysis_active = False

    if run_analysis:
        st.session_state.analysis_active = True

    if st.session_state.analysis_active:
        n_total = len(data)
        mean = data.mean()
        
        # --- [추가/수정됨] Minitab 방식 표준편차 계산 로직 ---
        std_overall = data.std(ddof=1) # 장기 표준편차 (전체)
        
        # 단기 표준편차 (군내 변동) 추정
        n_sub = subgroup_size
        std_within = std_overall # 기본값 (계산 실패 대비)
        
        if n_sub == 1 and n_total > 1:
            # Moving Range (이동범위) 방식
            mr = np.abs(np.diff(data))
            mr_bar = np.mean(mr)
            std_within = mr_bar / D2_TABLE.get(2, 1.128)
        elif n_sub > 1 and n_total >= n_sub:
            # R-bar (부분군 범위 평균) 방식
            num_subgroups = n_total // n_sub
            subgroup_data = data.values[:num_subgroups * n_sub].reshape(-1, n_sub)
            ranges = subgroup_data.max(axis=1) - subgroup_data.min(axis=1)
            r_bar = ranges.mean()
            d2_val = D2_TABLE.get(n_sub, 3.0) 
            std_within = r_bar / d2_val
            
        # 공정능력(Cp, Cpk, Pp, Ppk) 계산 통합 함수
        def calc_capability(std_val):
            cpu = (usl - mean) / (3 * std_val) if (usl is not None and std_val > 0) else None
            cpl = (mean - lsl) / (3 * std_val) if (lsl is not None and std_val > 0) else None
            
            if cpu is not None and cpl is not None:
                cx = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
                cxk = min(cpu, cpl)
            elif cpu is not None:
                cx, cxk = None, cpu
            elif cpl is not None:
                cx, cxk = None, cpl
            else:
                cx, cxk = None, None
                
            sigma = 3 * cxk if cxk is not None else None
            return cx, cxk, sigma

        cp, cpk, sigma_within = calc_capability(std_within)
        pp, ppk, sigma_overall = calc_capability(std_overall)
        
        # 빈칸(None)일 경우 "N/A" 처리 함수
        def fmt(val, dec=3): return f"{val:.{dec}f}" if val is not None else "N/A"

        st.markdown(f"## 📋 {column_name} 분석 요약 지표 (Minitab Std.)")
        
        st.write("**[ 단기 공정능력 (Within) - 이동범위/부분군 반영 ]**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("샘플 수 (N)", f"{n_total}")
        c2.metric("평균 (Mean)", fmt(mean, 3))
        c3.metric("StDev (Within)", fmt(std_within, 4))
        c4.metric("Cpk (단기능력)", fmt(cpk, 2))
        c5.metric("Sigma Level", f"{sigma_within:.2f}σ" if sigma_within is not None else "N/A")

        st.write("**[ 장기 공정능력 (Overall) - 전체 산포 반영 ]**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("부분군 크기 (n)", f"{n_sub}")
        c2.metric("LSL / USL", f"{fmt(lsl, 1)} / {fmt(usl, 1)}")
        c3.metric("StDev (Overall)", fmt(std_overall, 4))
        c4.metric("Ppk (장기성능)", fmt(ppk, 2))
        c5.metric("Sigma Level", f"{sigma_overall:.2f}σ" if sigma_overall is not None else "N/A")

        tab1, tab2, tab3 = st.tabs(["📊 공정능력 리포트", "📈 관리도", "📋 정규성 검정 리포트"])

        with tab1:
            st.subheader("Process Capability Histogram")
            if x_axis_mode == "자동 (Auto)":
                d_min, d_max = data.min(), data.max()
                d_range = d_max - d_min
                raw_step = d_range / 15
                magnitude = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1
                res = raw_step / magnitude
                if res <= 1.5: pretty_step = 1.0 * magnitude
                elif res <= 3.0: pretty_step = 2.0 * magnitude
                elif res <= 7.0: pretty_step = 5.0 * magnitude
                else: pretty_step = 10.0 * magnitude
                
                bin_size = pretty_step
                start_val = np.floor(d_min / bin_size) * bin_size - (bin_size / 2)
                
                plot_min_val = min(d_min, lsl) if lsl is not None else d_min
                plot_max_val = max(d_max, usl) if usl is not None else d_max
                
                plot_min = np.floor(plot_min_val / bin_size) * bin_size - bin_size
                plot_max = np.ceil(plot_max_val / bin_size) * bin_size + bin_size
                x_range_vals = [plot_min, plot_max]
                display_dtick = bin_size
                
                st.session_state.auto_x_min = float(plot_min)
                st.session_state.auto_x_max = float(plot_max)
                st.session_state.auto_x_step = float(bin_size)
            else:
                x_range_vals = [x_min_val, x_max_val]
                bin_size = x_step
                start_val = (np.floor(data.min() / bin_size) * bin_size) - (bin_size / 2)
                display_dtick = x_step

            # 정규분포 곡선 (미니탭처럼 Within StDev 기준으로 작도)
            x_curve = np.linspace(x_range_vals[0], x_range_vals[1], 500)
            y_pdf_within = norm.pdf(x_curve, mean, std_within) * n_total * bin_size

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, xbins=dict(start=start_val, size=bin_size),
                marker=dict(color='#D6EAF8', line=dict(color='#2E86C1', width=1)),
                name="Measured", hovertemplate="<b>중심: %{x:.3f}</b><br>Count: %{y}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(x=x_curve, y=y_pdf_within, mode='lines', line=dict(color='#1B4F72', width=3), 
                                     name="Normal (Within)", hovertemplate="Normal Dist: %{y:.2f}<extra></extra>"))

            # 가이드라인 (값이 있는 것만 추가 & Target 글자 겹침 방지 적용)
            guides = []
            if lsl is not None: guides.append((lsl, "LSL", "#E74C3C", 1.02))
            if usl is not None: guides.append((usl, "USL", "#E74C3C", 1.02))
            if target is not None: guides.append((target, "Target", "#7F8C8D", 1.07))
            guides.append((mean, "Mean", "#27AE60", 1.02))

            for val, name, color, y_pos in guides:
                fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5)
                fig.add_annotation(x=val, y=y_pos, yref="paper", text=f"<b>{name}</b>", 
                                   showarrow=False, font=dict(color=color, size=12), yanchor="bottom")

            if y_axis_mode == "자동 (Auto)":
                counts, _ = np.histogram(data, bins=np.arange(start_val, data.max() + bin_size*2, bin_size))
                y_max_auto = max(np.max(counts), np.max(y_pdf_within)) * 1.15
                
                st.session_state.auto_y_min = 0.0
                st.session_state.auto_y_max = float(y_max_auto)
                
                raw_y_step = y_max_auto / 10
                mag_y = 10 ** np.floor(np.log10(raw_y_step)) if raw_y_step > 0 else 1
                res_y = raw_y_step / mag_y
                if res_y <= 2: pretty_y_step = 2.0 * mag_y
                elif res_y <= 5: pretty_y_step = 5.0 * mag_y
                else: pretty_y_step = 10.0 * mag_y
                st.session_state.auto_y_step = float(pretty_y_step)

                y_axis_setup = dict(title=y_axis_title, showgrid=True, gridcolor='#F2F3F4', range=[0, y_max_auto], autorange=False, rangemode="nonnegative")
            else:
                y_axis_setup = dict(title=y_axis_title, showgrid=True, gridcolor='#F2F3F4', range=[y_min_val, y_max_val], dtick=y_step, autorange=False)

            fig.update_layout(
                title=dict(text=f"Process Capability Report for {column_name}", x=0.5, xanchor='center', font=dict(size=24)),
                template="simple_white", hovermode="x",
                xaxis=dict(title=x_axis_title, dtick=display_dtick, range=x_range_vals, showgrid=True, gridcolor='#F2F3F4'),
                yaxis=y_axis_setup, 
                width=1200, height=650, margin=dict(l=60, r=220, t=120, b=60), showlegend=False
            )
            
            summary_items = [
                {"label": "Process Data", "is_header": True},
                {"label": "LSL", "value": fmt(lsl, 3)},
                {"label": "Target", "value": fmt(target, 3)},
                {"label": "USL", "value": fmt(usl, 3)},
                {"label": "Sample N", "value": f"{n_total}"},
                {"label": "Mean", "value": fmt(mean, 3)},
                {"label": "", "value": "", "is_header": False},
                {"label": "Within (단기)", "is_header": True},
                {"label": "StDev", "value": fmt(std_within, 4)},
                {"label": "Cp", "value": fmt(cp, 2)},
                {"label": "Cpk", "value": fmt(cpk, 2)},
                {"label": "", "value": "", "is_header": False},
                {"label": "Overall (장기)", "is_header": True},
                {"label": "StDev", "value": fmt(std_overall, 4)},
                {"label": "Pp", "value": fmt(pp, 2)},
                {"label": "Ppk", "value": fmt(ppk, 2)},
            ]
            add_interactive_summary_box(fig, summary_items, fig_height=650)

            st.plotly_chart(fig, use_container_width=False, config={'toImageButtonOptions': {'filename': f'Process_Capability_{column_name}'}})

        with tab2:
            st.subheader("Control Chart", help="관리도 시료군 크기가 1이면 에러 방지를 위해 Xbar-R 차트는 그려지지 않습니다.")
            
            n = subgroup_size
            if n > 1 and len(data) >= n * 2: 
                num_subgroups = len(data) // n
                subgroup_data = data.values[:num_subgroups * n].reshape(-1, n)
                
                x_bars = subgroup_data.mean(axis=1)
                ranges = subgroup_data.max(axis=1) - subgroup_data.min(axis=1)
                subgroup_indices = np.arange(1, num_subgroups + 1)
                
                a2, d4, d3 = 3/np.sqrt(n), 2.114, 0
                if n in D2_TABLE:
                    factors = {2:(1.880,3.267,0), 3:(1.023,2.574,0), 4:(0.729,2.282,0), 5:(0.577,2.114,0), 6:(0.483,2.004,0), 7:(0.419,1.924,0.076), 8:(0.373,1.864,0.136), 9:(0.337,1.816,0.184), 10:(0.308,1.777,0.223)}
                    a2, d4, d3 = factors.get(n)
                    
                x_double_bar = x_bars.mean()
                r_bar = ranges.mean()
                
                ucl_x, lcl_x = x_double_bar + a2 * r_bar, x_double_bar - a2 * r_bar
                ucl_r, lcl_r = d4 * r_bar, d3 * r_bar
                
                x_colors = ['red' if (x > ucl_x or x < lcl_x) else '#2E86C1' for x in x_bars]
                r_colors = ['red' if (r > ucl_r or r < lcl_r) else '#2E86C1' for r in ranges]
        
                from plotly.subplots import make_subplots
                fig_c = make_subplots(rows=2, cols=1, subplot_titles=("<b>Xbar Chart</b>", "<b>R Chart</b>"), vertical_spacing=0.20, shared_xaxes=False)
                
                fig_c.add_trace(go.Scatter(x=subgroup_indices, y=x_bars, mode='lines+markers', line=dict(color='#2E86C1', width=1.5), marker=dict(color=x_colors, size=6), name="Xbar"), row=1, col=1)
                fig_c.add_trace(go.Scatter(x=subgroup_indices, y=ranges, mode='lines+markers', line=dict(color='#2E86C1', width=1.5), marker=dict(color=r_colors, size=6), name="Range"), row=2, col=1)
                
                def add_cl(fig, val, name, color, dash, row):
                    fig.add_hline(y=val, line_dash=dash, line_color=color, line_width=1.5, row=row, col=1)
                    fig.add_annotation(xref="paper", x=1.01, y=val, yref=f"y{row}" if row==1 else "y2", text=f"<b>{name}</b>", showarrow=False, font=dict(color=color, size=12), xanchor="left", yanchor="middle")

                add_cl(fig_c, ucl_x, "UCL", "red", "dash", 1)
                add_cl(fig_c, x_double_bar, "Mean", "green", "dash", 1)
                add_cl(fig_c, lcl_x, "LCL", "red", "dash", 1)
                add_cl(fig_c, ucl_r, "UCL", "red", "dash", 2)
                add_cl(fig_c, r_bar, "Rbar", "green", "dash", 2)
                add_cl(fig_c, lcl_r, "LCL", "red", "dash", 2)

                fig_c.update_layout(title=dict(text=f"<b>Control Charts for {column_name}</b>", x=0.5, xanchor='center', font=dict(size=24)), template="simple_white", height=750, width=1200, showlegend=False, margin=dict(l=60, r=220, t=100, b=80), hovermode="x unified")
                
                summary_items = [
                    {"label": "Control Chart", "is_header": True},
                    {"label": "Subgroup Size", "value": f"{n}"},
                    {"label": "Sample N", "value": f"{len(data)}"},
                    {"label": "", "value": "", "is_header": False},
                    {"label": "Xbar Limits", "is_header": True},
                    {"label": "UCL", "value": f"{ucl_x:.3f}"},
                    {"label": "Mean", "value": f"{x_double_bar:.3f}"},
                    {"label": "LCL", "value": f"{lcl_x:.3f}"},
                    {"label": "", "value": "", "is_header": False},
                    {"label": "R Limits", "is_header": True},
                    {"label": "UCL", "value": f"{ucl_r:.3f}"},
                    {"label": "R-bar", "value": f"{r_bar:.3f}"},
                    {"label": "LCL", "value": f"{lcl_r:.3f}"}
                ]
                add_interactive_summary_box(fig_c, summary_items, x_pos=1.06, fig_height=750)
                st.plotly_chart(fig_c, use_container_width=False, config={'toImageButtonOptions': {'filename': f'Control_Chart_{column_name}'}})
            else:
                st.warning("⚠️ 부분군 크기(Subgroup Size)가 1이거나 데이터가 부족하여 Xbar-R 관리도를 생성할 수 없습니다.")

        with tab3:
            st.subheader("Probability Plot (Normality Test)")
            sorted_data = np.sort(data)
            
            perc = (np.arange(1, n_total + 1) - 0.375) / (n_total + 0.25)
            theoretical_q = stats.norm.ppf(perc)
            
            ad_result = stats.anderson(data, dist='norm')
            ad_stat = ad_result.statistic
            p_val = stats.shapiro(data).pvalue

            slope, intercept, _, _, _ = stats.linregress(sorted_data, theoretical_q)
            
            prob_min, prob_max = 0.001, 0.999 
            y_min_limit, y_max_limit = stats.norm.ppf(prob_min), stats.norm.ppf(prob_max)
            x_start, x_end = (y_min_limit - intercept) / slope, (y_max_limit - intercept) / slope
            
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(x=sorted_data, y=theoretical_q, mode='markers', marker=dict(color='#2E86C1', size=6, symbol='circle-open'), name="Data", hovertemplate="<b>Value: %{x:.3f}</b><br>Percent: %{customdata:.1f}%<extra></extra>", customdata=perc*100))
            fig_norm.add_trace(go.Scatter(x=np.array([x_start, x_end]), y=slope * np.array([x_start, x_end]) + intercept, mode='lines', line=dict(color='#E74C3C', width=2), name="Fit Line", hoverinfo='skip'))

            tick_probs = [0.001, 0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 0.999]
            tick_vals = stats.norm.ppf(tick_probs)
            tick_text = [f"{p*100:.1f}" if (p*100 < 1 or p*100 > 99) else f"{p*100:.0f}" for p in tick_probs]

            fig_norm.update_layout(title=dict(text=f"<b>Probability Plot of {column_name}</b>", x=0.5, xanchor='center', font=dict(size=24)), template="simple_white", width=1200, height=650, margin=dict(l=60, r=220, t=100, b=60), showlegend=False, xaxis=dict(title=x_axis_title, showgrid=True, gridcolor='#F2F3F4'), yaxis=dict(title="Percent", tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[y_min_limit, y_max_limit], showgrid=True, gridcolor='#F2F3F4', zeroline=False))

            summary_items = [
                {"label": "Normality Test", "is_header": True},
                {"label": "AD Stat", "value": f"{ad_stat:.3f}"},
                {"label": "P-Value", "value": f"{p_val:.4f}"},
                {"label": "", "value": "", "is_header": False},
                {"label": "Stats", "is_header": True},
                {"label": "Mean", "value": fmt(mean, 3)},
                {"label": "StDev", "value": fmt(std_overall, 3)},
                {"label": "Sample N", "value": f"{n_total}"},
            ]
            add_interactive_summary_box(fig_norm, summary_items, fig_height=650)
            st.plotly_chart(fig_norm, use_container_width=False, config={'toImageButtonOptions': {'filename': f'Probability_Plot_{column_name}'}})    
            
        st.write("---")
        
        summary_data = [
            {"Report": "Process Capability", "Category": "Specs", "Metric": "LSL", "Value": fmt(lsl, 3)},
            {"Report": "Process Capability", "Category": "Specs", "Metric": "Target", "Value": fmt(target, 3)},
            {"Report": "Process Capability", "Category": "Specs", "Metric": "USL", "Value": fmt(usl, 3)},
            {"Report": "Process Capability", "Category": "Data", "Metric": "Sample N", "Value": f"{n_total}"},
            {"Report": "Process Capability", "Category": "Data", "Metric": "Mean", "Value": fmt(mean, 3)},
            {"Report": "Process Capability", "Category": "Within (Short-Term)", "Metric": "StDev (Within)", "Value": fmt(std_within, 4)},
            {"Report": "Process Capability", "Category": "Within (Short-Term)", "Metric": "Cp", "Value": fmt(cp, 2)},
            {"Report": "Process Capability", "Category": "Within (Short-Term)", "Metric": "Cpk", "Value": fmt(cpk, 2)},
            {"Report": "Process Capability", "Category": "Overall (Long-Term)", "Metric": "StDev (Overall)", "Value": fmt(std_overall, 4)},
            {"Report": "Process Capability", "Category": "Overall (Long-Term)", "Metric": "Pp", "Value": fmt(pp, 2)},
            {"Report": "Process Capability", "Category": "Overall (Long-Term)", "Metric": "Ppk", "Value": fmt(ppk, 2)},
            {"Report": "Normality Test", "Category": "Test Result", "Metric": "AD Stat", "Value": f"{ad_stat:.3f}"},
            {"Report": "Normality Test", "Category": "Test Result", "Metric": "P-Value", "Value": f"{p_val:.4f}"}
        ]
        
        df_res = pd.DataFrame(summary_data)
        csv_summary = df_res.to_csv(index=False)
        df_raw = pd.DataFrame({f"Raw Data ({column_name})": data.values})
        csv_raw = df_raw.to_csv(index=False)
        final_csv_str = csv_summary + "\n\n" + csv_raw
        csv_bytes = final_csv_str.encode('utf-8-sig')
        
        st.download_button(
            label=f"📥 {column_name} 통합 분석 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=f"Process_Analysis_{column_name}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 상단의 업로드 박스에 데이터를 넣고 [Process Capability Analysis Start] 버튼을 눌러주세요.")
