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
    "StDev": "표준편차 (Standard Deviation)<br>데이터가 평균으로부터 흩어진 정도를 나타냅니다.",
    "Cp": "공정능력지수 (Process Capability)<br>치우침을 고려하지 않은 공정의 잠재적 능력입니다.<br>((USL - LSL) / 6σ)",
    "Cpk": "실제 공정능력지수 (Process Capability Index)<br>데이터 평균의 치우침을 반영한 실제 공정 능력입니다.<br>(min(Cpu, Cpl))",
    "Sigma Level": "시그마 수준 (Sigma Level)<br>공정의 불량률을 나타내는 지표로, 높을수록 불량이 적습니다.<br>(3 × Cpk)",
    "Subgroup Size": "부분군 크기 (Subgroup Size)<br>한 번 샘플링할 때 묶는 데이터의 개수입니다.",
    "Total Points": "전체 데이터 수<br>관리도에 타점된 총 점의 개수입니다.",
    "UCL": "관리상한 (Upper Control Limit)<br>공정의 우연 원인에 의한 자연스러운 변동의 상한선입니다.",
    "CL": "중심선 (Center Line)<br>공정 데이터의 평균적인 수준을 나타내는 기준선입니다.",
    "LCL": "관리하한 (Lower Control Limit)<br>공정의 우연 원인에 의한 자연스러운 변동의 하한선입니다.",
    "R-bar": "범위 평균 (Average Range)<br>각 부분군 내의 범위(최댓값-최솟값)들의 평균입니다.",
    "AD Stat": "Anderson-Darling 통계량<br>데이터가 정규분포를 따르는지 검정하는 수치입니다.<br>값이 작을수록 정규분포에 가깝습니다.",
    "P-Value": "유의확률 (P-Value)<br>정규성 검정의 판단 기준입니다.<br>0.05 이상이면 정규분포를 따른다고 판단합니다."
}

def add_interactive_summary_box(fig, lines, x_pos=1.02, y_center=0.5, fig_height=650):
    """
    그래프 높이(px)가 달라도 글자 간격(px)을 절대적으로 고정하여
    모든 탭에서 시각적으로 완벽하게 동일한 밀도를 유지하는 함수
    """
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

# 세션 상태 초기화
if 'current_col' not in st.session_state:
    st.session_state.current_col = None
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

st.markdown("""
    <style>
    div[data-baseweb="popover"] {
        min-width: 500px !important;
        max-width: 800px !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("📊 Process Capability Analysis v0.2")

# 3. 데이터 가이드 섹션
with st.expander("ℹ️ 데이터 입력 형식 가이드 & 예시 파일 다운로드 (Click)", expanded=False):
    st.markdown("""
    ### 📂 데이터 준비 방법
    분석 정확도를 위해 아래 형식을 권장합니다.
    1. **첫 번째 행(Header):** 데이터의 이름(예: `Length`, `Weight`)을 적어주세요.
    2. **두 번째 행부터:** 실제 측정값(숫자)만 입력해주세요.
    """)
    
    example_df = pd.DataFrame({
        "Length (예시)": [0.402, 0.405, 0.398, 0.410, 0.401, "...", 0.403],
        "Weight (예시)": [10.5, 10.2, 10.8, 10.4, 10.6, "...", 10.5]
    })
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.write("#### 👀 데이터 미리보기")
        st.dataframe(example_df, hide_index=True, use_container_width=True)
    
    with c2:
        st.write("#### 💾 샘플 파일 다운로드")
        st.write("테스트용 샘플 데이터를 다운로드해서 바로 분석해보세요.")
        
        sample_csv_df = pd.DataFrame({
            "Measurement_Data": [0.426, 0.452, 0.413, 0.426, 0.413, 0.387, 0.452, 0.452, 0.401] * 10
        })
        csv_sample = sample_csv_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📥 예시 CSV 파일 다운로드",
            data=csv_sample,
            file_name="Sample_Data.csv",
            mime="text/csv",
            type="primary"
        )

# 2. 사이드바 설정
with st.sidebar:
    st.header("⚙️ 분석 설정")
    input_method = st.radio("데이터 입력 방식", ["파일 백업", "데이터 붙여넣기"])
    
    st.write("---")
    st.subheader("📏 규격치 (Specs)")
    st.caption("※ 하한이나 상한이 없는 경우, 숫자를 완전히 지워 빈칸으로 두세요.")
    
    # 1. 일반 텍스트 입력창으로 변경하여 빈칸 입력을 강제로 허용합니다.
    lsl_str = st.text_input("하한규격 (LSL)", value="0.150")
    target_str = st.text_input("목표치 (Target)", value="0.450")
    usl_str = st.text_input("상한규격 (USL)", value="0.750")
    
    # 2. 텍스트를 숫자로 변환하는 함수 (빈칸이거나 문자가 섞이면 None 반환)
    def parse_spec(val_str):
        if not val_str.strip():
            return None
        try:
            return float(val_str.strip())
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
        x_min_default = lsl - 0.05 if lsl is not None else 0.0
        x_max_default = usl + 0.05 if usl is not None else 1.0
        x_min_val = st.number_input("X축 최소값", value=st.session_state.get('auto_x_min', x_min_default), format="%.3f")
        x_max_val = st.number_input("X축 최대값", value=st.session_state.get('auto_x_max', x_max_default), format="%.3f")
        x_step = st.number_input("X축 눈금 단위 (Bin Size)", value=st.session_state.get('auto_x_step', 0.020), format="%.3f", min_value=0.001)
    
    x_axis_title = st.text_input("X축 제목", value="Measurement Value")

    st.write("**[Y축 설정 (Histogram)]**")
    y_axis_mode = st.radio("Y축 범위 모드", ["자동 (Auto)", "수동 (Manual)"])
    if y_axis_mode == "수동 (Manual)":
        y_min_val = st.number_input("Y축 최소값", value=st.session_state.get('auto_y_min', 0.0), format="%.1f")
        y_max_val = st.number_input("Y축 최대값", value=st.session_state.get('auto_y_max', 500.0), format="%.1f")
        y_step = st.number_input("Y축 눈금 단위", value=st.session_state.get('auto_y_step', 50.0), format="%.1f", min_value=0.1)
        
    y_axis_title = st.text_input("Y축 제목", value="Frequency")    
    
    st.write("---")
    subgroup_size = st.number_input("관리도 시료군(n) 크기", value=5, min_value=1)

# 3. 메인 화면 - 데이터 로드 로직
data = pd.Series(dtype=float)
column_name = ""

if input_method == "파일 백업":
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
        st.warning(f"⚠️ 분석 대상이 '{column_name}'으로 변경되었습니다. 버튼을 눌러 분석을 시작하세요.")

    if run_analysis:
        st.session_state.analysis_active = True

    if st.session_state.analysis_active:
        # 기초 통계 계산
        mean, std = data.mean(), data.std(ddof=1)
        
        # 한쪽 규격만 있는 경우 처리
        cpu = (usl - mean) / (3 * std) if (usl is not None and std > 0) else None
        cpl = (mean - lsl) / (3 * std) if (lsl is not None and std > 0) else None
        
        if cpu is not None and cpl is not None:
            cp = (usl - lsl) / (6 * std) if std > 0 else 0
            cpk = min(cpu, cpl)
        elif cpu is not None:
            cp = None 
            cpk = cpu
        elif cpl is not None:
            cp = None
            cpk = cpl
        else:
            cp = None
            cpk = None
            
        sigma_lvl = 3 * cpk if cpk is not None else None
        
        # 규격 정합성 체크 (양쪽 규격 모두 있을 때만)
        if lsl is not None and usl is not None:
            spec_range = usl - lsl
            if not (lsl - spec_range < mean < usl + spec_range):
                st.error(f"❌ 규격({lsl}~{usl})과 데이터 평균({mean:.3f})의 차이가 너무 큽니다. 규격 설정을 확인해주세요.")
                
        # 빈칸(None)일 경우 "N/A" 처리 함수
        def fmt(val, dec=3): return f"{val:.{dec}f}" if val is not None else "N/A"

        st.markdown(f"## 📋 {column_name} 분석 요약 지표")
        m_cols = st.columns(6)
        m_cols[0].metric("샘플 수 (N)", f"{len(data)}")
        m_cols[1].metric("평균 (Mean)", fmt(mean, 3))
        m_cols[2].metric("표준편차 (σ)", fmt(std, 3))
        m_cols[3].metric("Cp", fmt(cp, 2))
        m_cols[4].metric("Cpk", fmt(cpk, 2))
        m_cols[5].metric("Sigma Level", f"{sigma_lvl:.2f}σ" if sigma_lvl is not None else "N/A")

        tab1, tab2, tab3 = st.tabs(["📊 공정능력 리포트", "📈 관리도", "📋 정규성 검정 리포트"])

        with tab1:
            st.subheader("Process Capability Histogram", help="""
**📊 공정능력 리포트란?**
현재 공정이 고객이 요구하는 규격(LSL~USL) 내에서 제품을 얼마나 잘 생산할 수 있는지 보여줍니다.
""")
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
                
                # 규격이 없을 경우 데이터 최솟값/최댓값을 기준
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

            x_curve = np.linspace(x_range_vals[0], x_range_vals[1], 500)
            y_pdf = norm.pdf(x_curve, mean, std) * len(data) * bin_size

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, xbins=dict(start=start_val, size=bin_size),
                marker=dict(color='#D6EAF8', line=dict(color='#2E86C1', width=1)),
                name="Measured", hovertemplate="<b>중심: %{x:.3f}</b><br>Count: %{y}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(x=x_curve, y=y_pdf, mode='lines', line=dict(color='#1B4F72', width=3), 
                                     name="Normal", hovertemplate="Normal Dist: %{y:.2f}<extra></extra>"))

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
                y_max_auto = max(np.max(counts), np.max(y_pdf)) * 1.15
                
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
                {"label": "Sample N", "value": f"{len(data)}"},
                {"label": "Mean", "value": fmt(mean, 3)},
                {"label": "StDev", "value": fmt(std, 3)},
                {"label": "", "value": "", "is_header": False},
                {"label": "Capability", "is_header": True},
                {"label": "Cp", "value": fmt(cp, 2)},
                {"label": "Cpk", "value": fmt(cpk, 2)},
                {"label": "Sigma Level", "value": fmt(sigma_lvl, 2)},
            ]
            add_interactive_summary_box(fig, summary_items, fig_height=650)

            st.plotly_chart(fig, use_container_width=False, config={'toImageButtonOptions': {'filename': f'Process_Capability_{column_name}'}})

        with tab2:
            st.subheader("Xbar-R Control Chart", help="""
**📈 관리도(Control Chart)란?**
공정이 시간 흐름에 따라 통계적으로 안정된 상태(관리 상태)인지 확인하는 도구입니다.
""")
            factors = {
                2: (1.880, 3.267, 0),
                3: (1.023, 2.574, 0),
                4: (0.729, 2.282, 0),
                5: (0.577, 2.114, 0),
                6: (0.483, 2.004, 0),
                7: (0.419, 1.924, 0.076),
                8: (0.373, 1.864, 0.136),
                9: (0.337, 1.816, 0.184),
                10: (0.308, 1.777, 0.223) 
            }
            
            n = subgroup_size
            if len(data) >= n * 2: 
                num_subgroups = len(data) // n
                subgroup_data = data.values[:num_subgroups * n].reshape(-1, n)
                
                x_bars = subgroup_data.mean(axis=1)
                ranges = subgroup_data.max(axis=1) - subgroup_data.min(axis=1)
                subgroup_indices = np.arange(1, num_subgroups + 1)
                
                a2, d4, d3 = factors.get(n, (3/np.sqrt(n), 2.114, 0))
                x_double_bar = x_bars.mean()
                r_bar = ranges.mean()
                
                ucl_x, lcl_x = x_double_bar + a2 * r_bar, x_double_bar - a2 * r_bar
                ucl_r, lcl_r = d4 * r_bar, d3 * r_bar
                
                x_colors = ['red' if (x > ucl_x or x < lcl_x) else '#2E86C1' for x in x_bars]
                r_colors = ['red' if (r > ucl_r or r < lcl_r) else '#2E86C1' for r in ranges]
        
                from plotly.subplots import make_subplots
                fig_c = make_subplots(
                    rows=2, cols=1, 
                    subplot_titles=("<b>Xbar Chart</b>", "<b>R Chart</b>"), 
                    vertical_spacing=0.20, 
                    shared_xaxes=False     
                )
                
                fig_c.add_trace(go.Scatter(
                    x=subgroup_indices,
                    y=x_bars, 
                    mode='lines+markers', 
                    line=dict(color='#2E86C1', width=1.5), 
                    marker=dict(color=x_colors, size=6), 
                    name="Xbar",
                    hovertemplate="<b>Subgroup: %{x}</b><br>Mean: %{y:.3f}<extra></extra>"
                ), row=1, col=1)
                
                fig_c.add_trace(go.Scatter(
                    x=subgroup_indices,
                    y=ranges, 
                    mode='lines+markers', 
                    line=dict(color='#2E86C1', width=1.5), 
                    marker=dict(color=r_colors, size=6), 
                    name="Range",
                    hovertemplate="<b>Subgroup: %{x}</b><br>Range: %{y:.3f}<extra></extra>"
                ), row=2, col=1)
                
                def add_control_limit(fig, val, name, color, dash, row):
                    fig.add_hline(y=val, line_dash=dash, line_color=color, line_width=1.5, row=row, col=1)
                    fig.add_annotation(
                        xref="paper", x=1.01, 
                        y=val, yref=f"y{row}" if row==1 else "y2",
                        text=f"<b>{name}</b>",
                        showarrow=False,
                        font=dict(color=color, size=12),
                        xanchor="left", yanchor="middle"
                    )

                add_control_limit(fig_c, ucl_x, "UCL", "red", "dash", 1)
                add_control_limit(fig_c, x_double_bar, "Mean", "green", "dash", 1)
                add_control_limit(fig_c, lcl_x, "LCL", "red", "dash", 1)

                add_control_limit(fig_c, ucl_r, "UCL", "red", "dash", 2)
                add_control_limit(fig_c, r_bar, "Rbar", "green", "dash", 2)
                add_control_limit(fig_c, lcl_r, "LCL", "red", "dash", 2)

                fig_c.update_layout(
                    title=dict(text=f"<b>Control Charts for {column_name}</b>", x=0.5, xanchor='center', font=dict(size=24)),
                    template="simple_white", 
                    height=750, 
                    width=1200, 
                    showlegend=False, 
                    margin=dict(l=60, r=220, t=100, b=80), 
                    hovermode="x unified"
                )
                
                fig_c.update_yaxes(title_text="Sample Mean", showgrid=True, gridcolor='#F2F3F4', row=1, col=1)
                fig_c.update_xaxes(title_text="Subgroup Number", showgrid=True, gridcolor='#F2F3F4', row=1, col=1)
                
                fig_c.update_yaxes(title_text="Sample Range", showgrid=True, gridcolor='#F2F3F4', row=2, col=1)
                fig_c.update_xaxes(title_text="Subgroup Number", showgrid=True, gridcolor='#F2F3F4', row=2, col=1)
                
                summary_items = [
                    {"label": "Control Chart Stats", "is_header": True},
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
                st.warning("데이터 개수가 부족하여 Xbar-R 관리도를 생성할 수 없습니다.")

        with tab3:
            st.subheader("Probability Plot (Normality Test)", help="""
**📋 정규성 검정(Probability Plot)이란?**
수집된 데이터가 통계적으로 '정규분포(종 모양)'를 따르는지 검증합니다.
""")
            sorted_data = np.sort(data)
            n_total = len(data)
            
            perc = (np.arange(1, n_total + 1) - 0.375) / (n_total + 0.25)
            theoretical_q = stats.norm.ppf(perc)
            
            ad_result = stats.anderson(data, dist='norm')
            ad_stat = ad_result.statistic
            p_val = stats.shapiro(data).pvalue

            slope, intercept, r_val, _, _ = stats.linregress(sorted_data, theoretical_q)
            
            prob_min, prob_max = 0.001, 0.999 
            y_min_limit = stats.norm.ppf(prob_min)
            y_max_limit = stats.norm.ppf(prob_max)
            
            x_start = (y_min_limit - intercept) / slope
            x_end = (y_max_limit - intercept) / slope
            
            line_x = np.array([x_start, x_end])
            line_y = slope * line_x + intercept

            fig_norm = go.Figure()

            fig_norm.add_trace(go.Scatter(
                x=sorted_data, 
                y=theoretical_q, 
                mode='markers',
                marker=dict(color='#2E86C1', size=6, symbol='circle-open'), 
                name="Data",
                hovertemplate="<b>Value: %{x:.3f}</b><br>Percent: %{customdata:.1f}%<extra></extra>",
                customdata=perc*100 
            ))

            fig_norm.add_trace(go.Scatter(
                x=line_x, 
                y=line_y, 
                mode='lines',
                line=dict(color='#E74C3C', width=2),
                name="Fit Line",
                hoverinfo='skip'
            ))

            tick_probs = [0.001, 0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 0.999]
            tick_vals = stats.norm.ppf(tick_probs)
            tick_text = []
            for p in tick_probs:
                val = p * 100
                if val < 1 or val > 99:
                    tick_text.append(f"{val:.1f}")
                else:
                    tick_text.append(f"{val:.0f}")

            fig_norm.update_layout(
                title=dict(text=f"<b>Probability Plot of {column_name}</b>", x=0.5, xanchor='center', font=dict(size=24)),
                template="simple_white",
                width=1200, height=650,
                margin=dict(l=60, r=220, t=100, b=60),
                showlegend=False,
                xaxis=dict(title=x_axis_title, showgrid=True, gridcolor='#F2F3F4'),
                yaxis=dict(
                    title="Percent", 
                    tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                    range=[y_min_limit, y_max_limit], 
                    showgrid=True, gridcolor='#F2F3F4', zeroline=False
                )
            )

            summary_items = [
                {"label": "Normality Test", "is_header": True},
                {"label": "AD Stat", "value": f"{ad_stat:.3f}"},
                {"label": "P-Value", "value": f"{p_val:.4f}"},
                {"label": "", "value": "", "is_header": False},
                {"label": "Stats", "is_header": True},
                {"label": "Mean", "value": f"{mean:.3f}"},
                {"label": "StDev", "value": f"{std:.3f}"},
                {"label": "Sample N", "value": f"{len(data)}"},
            ]
            add_interactive_summary_box(fig_norm, summary_items, fig_height=650)
            
            st.plotly_chart(fig_norm, use_container_width=False, config={'toImageButtonOptions': {'filename': f'Probability_Plot_{column_name}'}})    
            
        st.write("---")
        
        summary_data = [
            {"Report": "Process Capability", "Category": "Specs", "Metric": "LSL", "Value": fmt(lsl, 3)},
            {"Report": "Process Capability", "Category": "Specs", "Metric": "Target", "Value": fmt(target, 3)},
            {"Report": "Process Capability", "Category": "Specs", "Metric": "USL", "Value": fmt(usl, 3)},
            {"Report": "Process Capability", "Category": "Process Data", "Metric": "Sample N", "Value": f"{len(data)}"},
            {"Report": "Process Capability", "Category": "Process Data", "Metric": "Mean", "Value": fmt(mean, 3)},
            {"Report": "Process Capability", "Category": "Process Data", "Metric": "StdDev", "Value": fmt(std, 3)},
            {"Report": "Process Capability", "Category": "Capability", "Metric": "Cp", "Value": fmt(cp, 2)},
            {"Report": "Process Capability", "Category": "Capability", "Metric": "Cpk", "Value": fmt(cpk, 2)},
            {"Report": "Process Capability", "Category": "Capability", "Metric": "Sigma Level", "Value": fmt(sigma_lvl, 2)},
            
            {"Report": "Control Chart", "Category": "Settings", "Metric": "Subgroup Size", "Value": f"{subgroup_size}"},
            {"Report": "Control Chart", "Category": "Xbar Limits", "Metric": "UCL", "Value": f"{ucl_x:.3f}"},
            {"Report": "Control Chart", "Category": "Xbar Limits", "Metric": "CL (Mean)", "Value": f"{x_double_bar:.3f}"},
            {"Report": "Control Chart", "Category": "Xbar Limits", "Metric": "LCL", "Value": f"{lcl_x:.3f}"},
            {"Report": "Control Chart", "Category": "R Limits", "Metric": "UCL", "Value": f"{ucl_r:.3f}"},
            {"Report": "Control Chart", "Category": "R Limits", "Metric": "CL (R-bar)", "Value": f"{r_bar:.3f}"},
            {"Report": "Control Chart", "Category": "R Limits", "Metric": "LCL", "Value": f"{lcl_r:.3f}"}, 

            {"Report": "Normality Test", "Category": "Test Result", "Metric": "AD Stat", "Value": f"{ad_stat:.3f}"},
            {"Report": "Normality Test", "Category": "Test Result", "Metric": "P-Value", "Value": f"{p_val:.4f}"}
        ]
        
        # 1. 요약 데이터를 CSV 문자열로 변환
        df_res = pd.DataFrame(summary_data)
        csv_summary = df_res.to_csv(index=False)
        
        # 2. 원본(Raw) 데이터를 데이터프레임으로 만들고 CSV 문자열로 변환
        df_raw = pd.DataFrame({f"Raw Data ({column_name})": data.values})
        csv_raw = df_raw.to_csv(index=False)
        
        # 3. 두 CSV 문자열을 결합 (중간에 빈 줄 삽입하여 엑셀에서 보기 좋게 분리)
        final_csv_str = csv_summary + "\n" + csv_raw
        
        # 한글 깨짐 방지를 위해 utf-8-sig로 인코딩
        csv_bytes = final_csv_str.encode('utf-8-sig')
        
        st.download_button(
            label=f"📥 {column_name} 통합 분석 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=f"Process_Analysis_{column_name}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 상단의 업로드 박스에 데이터를 넣고 [Process Capability Analysis Start] 버튼을 눌러주세요.")
