import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# ==============================
# 0. 页面基础配置 + 医疗科研风格
# ==============================
st.set_page_config(
    page_title="AECOPD风险智能预测平台",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071f45 0%, #0a2f68 52%, #07224a 100%);
        color: white;
        border-right: 2px solid rgba(120, 193, 255, 0.32);
        box-shadow: inset -10px 0 24px rgba(0, 0, 0, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2.1rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .sidebar-brand {
        background:
            radial-gradient(circle at 88% 18%, rgba(110, 206, 255, 0.35), transparent 44%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.17), rgba(255, 255, 255, 0.03));
        border: 1px solid rgba(140, 212, 255, 0.28);
        border-radius: 16px;
        padding: 16px 14px;
        margin-bottom: 20px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        max-width: 320px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    .sidebar-brand .name {
        font-size: 28px;
        font-weight: 900;
        letter-spacing: 0.4px;
        line-height: 1.2;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.28);
    }
    .sidebar-brand .tagline {
        margin-top: 10px;
        font-size: 14px;
        font-weight: 600;
        color: #cfeaff;
        opacity: 0.98;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 10px;
        max-width: 320px;
        margin-left: auto;
        margin-right: auto;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 12px;
        padding: 10px 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: background-color .15s ease, border-color .15s ease, transform .12s ease;
        width: 100%;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: rgba(118, 198, 255, 0.14);
        border-color: rgba(133, 208, 255, 0.48);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.2px;
        line-height: 1.2;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.2);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        width: 18px;
        height: 18px;
        accent-color: #ff4f4f;
        filter: drop-shadow(0 0 4px rgba(255, 79, 79, 0.35));
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 16px;
        margin-bottom: 18px;
        border-color: rgba(147, 215, 255, 0.2);
    }
    .main-title {
        color: #0b2c59;
        font-weight: 700;
        border-left: 6px solid #0b2c59;
        padding-left: 10px;
        margin-bottom: 12px;
    }
    .sub-card {
        background: white;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(11, 44, 89, 0.08);
        margin-bottom: 12px;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 12px;
        min-height: 96px;
    }
    .result-card .label {
        font-size: 15px;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 8px;
    }
    .result-card .value {
        font-size: 42px;
        font-weight: 700;
        color: #1f2937;
        line-height: 1;
        margin-bottom: 0;
    }
    .result-card .value.low-risk {
        color: #16a34a;
    }
    .result-card .value.mid-risk {
        color: #d97706;
    }
    .result-card .value.high-risk {
        color: #dc2626;
    }
    .result-card .value-small {
        font-size: 18px;
        font-weight: 400;
        color: #1f2937;
        line-height: 1.3;
        margin-bottom: 0;
    }
    .scroll-down-btn {
        position: fixed;
        right: 28px;
        bottom: 28px;
        z-index: 9999;
        background: #0b2c59;
        color: white !important;
        text-decoration: none !important;
        width: 46px;
        height: 46px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(11, 44, 89, 0.25);
    }
    .scroll-down-btn:hover {
        background: #174a8b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 主界面悬浮“向下滑动”按钮
st.markdown(
    """
    <a class="scroll-down-btn" href="#page-bottom" title="向下滑动">↓</a>
    """,
    unsafe_allow_html=True,
)


# ==============================
# 1. 文件加载函数
# ==============================
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_resource
def load_explainer(explainer_path: str):
    return joblib.load(explainer_path)


@st.cache_data
def load_data(data_path: str):
    return pd.read_csv(data_path)


def parse_report(report_path: Path):
    """从 xgb_report.txt 中提取最优参数和测试集指标。"""
    params = {}
    metrics = {}
    if not report_path.exists():
        return params, metrics

    lines = report_path.read_text(encoding="utf-8-sig").splitlines()
    in_params = False
    in_metrics = False

    for line in lines:
        line = line.strip()
        if line == "最优参数:":
            in_params = True
            in_metrics = False
            continue
        if line == "测试集指标:":
            in_params = False
            in_metrics = True
            continue
        if not line:
            continue

        if in_params and ":" in line:
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
        if in_metrics and ":" in line:
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()

    return params, metrics


def format_decimal_str(value: str, decimals: int = 2) -> str:
    """将可解析为数字的字符串格式化为固定小数位。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if num.is_integer():
        return str(int(num))
    return f"{num:.{decimals}f}"


def make_prediction_input():
    """构建临床输入控件，范围严格贴合题目给定区间。"""
    c1, c2, c3 = st.columns(3)
    with c1:
        pastae = st.selectbox("PastAE（既往AECOPD病史）", [0, 1, 2], index=0)
        infect = st.selectbox("Infect（近期感染）", [0, 1], index=0)
        age = st.number_input("Age（年龄）", min_value=50, max_value=85, value=65, step=1)
        cardio = st.selectbox("Cardio（合并心血管疾病）", [0, 1], index=0)
        bronchiect = st.selectbox("Bronchiect（合并支气管扩张症）", [0, 1], index=0)

    with c2:
        smoke = st.selectbox("Smoke（吸烟状态）", [0, 1, 2], index=0)
        airpollu = st.selectbox("AirPollu（空气污染程度）", [0, 1, 2], index=0)
        bmi = st.number_input("BMI（体质量指数）", min_value=16.0, max_value=30.0, value=22.0, step=0.1)
        diabetes = st.selectbox("Diabetes（合并糖尿病）", [0, 1], index=0)
        tempchange = st.selectbox("TempChange（温度变化暴露）", [0, 1], index=0)

    with c3:
        fev1 = st.number_input("FEV1（肺功能FEV1%预测值）", min_value=30.0, max_value=80.0, value=55.0, step=0.1)
        psych = st.selectbox("Psych（负性心理状态）", [0, 1, 2], index=0)
        fibrinogen = st.number_input("Fibrinogen（纤维蛋白原 g/L）", min_value=2.0, max_value=5.0, value=3.2, step=0.1)
        sgrq = st.number_input("SGRQ（呼吸问卷评分）", min_value=10, max_value=90, value=45, step=1)
        followup = st.selectbox("FollowUp（复诊规律）", [0, 1], index=0)

    raw = {
        "PastAE": pastae,
        "Smoke": smoke,
        "FEV1": fev1,
        "Infect": infect,
        "AirPollu": airpollu,
        "Age": age,
        "BMI": bmi,
        "Cardio": cardio,
        "Diabetes": diabetes,
        "Bronchiect": bronchiect,
        "TempChange": tempchange,
        "Psych": psych,
        "Fibrinogen": fibrinogen,
        "SGRQ": sgrq,
        "FollowUp": followup,
    }
    return raw


def build_model_input(raw_inputs: dict, model_features):
    """
    将原始临床输入编码成模型需要的特征列：
    连续/二分类直接赋值；PastAE/Smoke/AirPollu/Psych做独热编码。
    """
    x = pd.DataFrame(
        np.zeros((1, len(model_features)), dtype=float),
        columns=model_features,
    )
    # 连续与二分类
    direct_cols = [
        "FEV1",
        "Infect",
        "Age",
        "BMI",
        "Cardio",
        "Diabetes",
        "Bronchiect",
        "TempChange",
        "Fibrinogen",
        "SGRQ",
        "FollowUp",
    ]
    for col in direct_cols:
        if col in x.columns:
            x.loc[0, col] = raw_inputs[col]

    # 多分类独热编码
    for base in ["PastAE", "Smoke", "AirPollu", "Psych"]:
        col_name = f"{base}_{int(raw_inputs[base])}"
        if col_name in x.columns:
            x.loc[0, col_name] = 1
    return x


def risk_level(prob):
    if prob < 0.3:
        return "低风险"
    if prob <= 0.6:
        return "中风险"
    return "高风险"


def safe_show_image(image_path: Path, caption: str | None = None, max_side: int = 2200):
    """
    安全显示大图：对超大分辨率图片先缩放，再交给 Streamlit 展示，
    避免 PIL DecompressionBombError。
    """
    try:
        old_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            resized = img.copy()
            resized.thumbnail((max_side, max_side))
        Image.MAX_IMAGE_PIXELS = old_limit
        st.image(resized, caption=caption, use_container_width=True)
    except Exception as exc:
        Image.MAX_IMAGE_PIXELS = old_limit
        st.warning(f"图像加载失败：{image_path.name}（{exc}）")


# ==============================
# 2. 左侧边栏导航
# ==============================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <p class="name">中南大学湘雅二医院</p>
            <div class="tagline">AECOPD 智能评估系统</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "导航",
        ["predict", "explore", "model"],
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: {
            "predict": "🩺 预测平台",
            "explore": "📊 数据探索",
            "model": "🧠 模型信息",
        }[x],
    )


# ==============================
# 3. 资源加载
# ==============================
model_path = Path("xgb_model.pkl")
explainer_path = Path("shap_explainer.pkl")
data_path = Path("cleaned_data.csv")
raw_data_path = Path("dataset.csv")
report_path = Path("xgb_report.txt")

if not model_path.exists() or not explainer_path.exists() or not data_path.exists():
    st.error("缺少必要文件：请确保 xgb_model.pkl、shap_explainer.pkl、cleaned_data.csv 均已生成。")
    st.stop()

model = load_model(str(model_path))
explainer = load_explainer(str(explainer_path))
df = load_data(str(data_path))
raw_df = load_data(str(raw_data_path)) if raw_data_path.exists() else None
model_features = list(getattr(model, "feature_names_in_", [c for c in df.columns if c != "AECOPD_occur"]))
best_params, metric_info = parse_report(report_path)


# ==============================
# 4. 页面1：预测平台
# ==============================
if page == "predict":
    st.markdown('<h2 class="main-title">AECOPD风险智能预测平台</h2>', unsafe_allow_html=True)
    st.markdown('<div class="sub-card">请录入患者临床特征后点击“一键预测”，结果仅在点击后更新。</div>', unsafe_allow_html=True)

    # 使用 form 保证“只有点击按钮才更新结果”，避免输入变化触发刷新
    with st.form("predict_form"):
        inputs = make_prediction_input()
        submitted = st.form_submit_button("一键预测", type="primary", use_container_width=True)

    # 初始化状态
    if "pred_prob" not in st.session_state:
        st.session_state["pred_prob"] = None
        st.session_state["pred_level"] = None
        st.session_state["pred_explanation"] = None
        st.session_state["pred_inputs"] = None

    # 仅在按钮点击时进行预测与SHAP计算
    if submitted:
        x_input = build_model_input(inputs, model_features)
        prob = float(model.predict_proba(x_input)[:, 1][0])
        lvl = risk_level(prob)

        shap_values = explainer.shap_values(x_input)
        if isinstance(shap_values, list):
            shap_row = np.array(shap_values[1])[0] if len(shap_values) > 1 else np.array(shap_values[0])[0]
        elif hasattr(shap_values, "values"):
            shap_row = np.array(shap_values.values)[0]
        else:
            shap_row = np.array(shap_values)[0]

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_value = explainer.expected_value

        one_exp = shap.Explanation(
            values=shap_row,
            base_values=base_value,
            data=x_input.iloc[0].values,
            feature_names=x_input.columns.tolist(),
        )

        st.session_state["pred_prob"] = prob
        st.session_state["pred_level"] = lvl
        st.session_state["pred_explanation"] = one_exp
        st.session_state["pred_inputs"] = inputs.copy()

    # 展示最近一次点击预测后的结果
    if st.session_state["pred_prob"] is not None:
        risk_css_class = {
            "低风险": "low-risk",
            "中风险": "mid-risk",
            "高风险": "high-risk",
        }.get(st.session_state["pred_level"], "")

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"""
            <div class="result-card">
                <div class="label">AECOPD发生风险概率</div>
                <p class="value {risk_css_class}">{st.session_state['pred_prob']:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""
            <div class="result-card">
                <div class="label">风险等级</div>
                <p class="value {risk_css_class}">{st.session_state["pred_level"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with c3:
            st.markdown(
                """
                <div class="result-card">
                    <div class="label">判定阈值</div>
                    <p class="value-small">低风险：&lt;0.3  |  中风险：0.3-0.6  |  高风险：&gt;0.6</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 个体解释：SHAP瀑布图（当前预测患者）")
        st.caption("说明：下图基于你最近一次点击“一键预测”时提交的输入值。若修改了输入，请再次点击按钮更新结果。")
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(st.session_state["pred_explanation"], show=False, max_display=15)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("请先点击“一键预测”以生成当前患者的风险结果与SHAP瀑布图。")


# ==============================
# 5. 页面2：数据探索
# ==============================
elif page == "explore":
    st.markdown('<h2 class="main-title">数据说明</h2>', unsafe_allow_html=True)

    st.markdown("### 数据集概况")
    st.write(f"- 样本量：**{df.shape[0]}**")
    st.write("- 研究目标：基于临床特征预测患者是否发生 AECOPD。")
    st.write("- 数据来源：某某医院临床科研数据（脱敏后用于建模分析）。")

    st.markdown("### 输入特征释义")
    feature_desc = pd.DataFrame(
        [
            ["PastAE", "既往AECOPD病史", "0/1/2", "0=无，1=1次，2=≥2次"],
            ["Smoke", "吸烟状态", "0/1/2", "0=不吸烟，1=既往吸烟，2=现吸烟"],
            ["FEV1", "肺功能FEV1%预测值", "30~80", "连续变量（%）"],
            ["Infect", "近期感染", "0/1", "0=无，1=有"],
            ["AirPollu", "空气污染程度", "0/1/2", "0=低，1=中，2=高"],
            ["Age", "年龄", "50~85", "连续变量（岁）"],
            ["BMI", "体质量指数", "16~30", "连续变量"],
            ["Cardio", "合并心血管疾病", "0/1", "0=无，1=有"],
            ["Diabetes", "合并糖尿病", "0/1", "0=无，1=有"],
            ["Bronchiect", "合并支气管扩张症", "0/1", "0=无，1=有"],
            ["TempChange", "温度变化暴露", "0/1", "0=少，1=多"],
            ["Psych", "负性心理状态", "0/1/2", "0=无，1=轻度，2=中重度"],
            ["Fibrinogen", "纤维蛋白原", "2.0~5.0", "连续变量（g/L）"],
            ["SGRQ", "SGRQ-C呼吸问卷评分", "10~90", "分数越高健康越差"],
            ["FollowUp", "复诊规律", "0/1", "0=规律，1=不规律"],
        ],
        columns=["变量名", "临床含义", "取值/单位", "编码说明"],
    )
    st.dataframe(feature_desc, use_container_width=True, hide_index=True)

    st.markdown("### 结局变量定义")
    st.write("- `AECOPD_occur`：0=未发生，1=发生（模型预测目标）。")

    st.markdown("### 数据预处理说明")
    st.write("- 缺失值处理：连续变量中位数填充，分类变量众数填充。")
    st.write("- 异常值处理：连续变量采用IQR方法并进行边界修正（clip）。")
    st.write("- 编码方式：多分类变量采用独热编码（One-Hot，1/0）。")

    st.markdown("### 原始数据（dataset.csv）")
    if raw_df is not None:
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.warning("未找到 dataset.csv，当前无法展示原始数据。")

    st.markdown("### 特征相关性热力图")
    heatmap_path = Path("corr_heatmap.png")
    if heatmap_path.exists():
        safe_show_image(heatmap_path, caption="清洗与编码后特征相关性热力图")
    else:
        st.warning("未找到 corr_heatmap.png，请先运行第一步脚本。")


# ==============================
# 6. 页面3：模型信息
# ==============================
else:
    st.markdown('<h2 class="main-title">模型说明</h2>', unsafe_allow_html=True)

    st.markdown("### 模型构建流程")
    st.write(
        "- 使用 XGBoostClassifier 构建二分类模型，以 `binary:logistic` 为目标函数，`AUC` 作为优化与评估核心指标。"
    )
    st.write("- 采用贝叶斯优化（5折交叉验证）搜索最优超参数，并结合 early stopping 防止过拟合。")

    st.markdown("### 最优模型参数")
    if best_params:
        formatted_params = {k: format_decimal_str(v, decimals=2) for k, v in best_params.items()}
        st.json(formatted_params)
    else:
        st.info("未读取到 xgb_report.txt 中的参数信息。")

    st.markdown("### 模型性能评估指标")
    if metric_info:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC", metric_info.get("AUC", "-"))
        c2.metric("Accuracy", metric_info.get("Accuracy", "-"))
        c3.metric("Recall", metric_info.get("Recall", "-"))
        c4.metric("Precision", metric_info.get("Precision", "-"))
        c5.metric("F1-score", metric_info.get("F1-score", "-"))
    else:
        st.info("未读取到 xgb_report.txt 中的指标信息。")

    st.markdown("### 模型可解释性（SHAP）")
    st.write("以下图像均来自第二步输出文件，便于临床医生和科研人员直观理解模型决策逻辑。")

    img_map = [
        ("shap_summary.png", "SHAP汇总图（特征重要性）", "表示整体样本中各特征对风险预测贡献的大小和方向，越靠前通常越重要。"),
        ("pastae_shap_dependence.png", "SHAP依赖图：PastAE", "观察既往AECOPD病史水平变化时，对风险概率的推动或抑制趋势。"),
        ("fev1_shap_dependence.png", "SHAP依赖图：FEV1", "反映肺功能指标变化与风险贡献之间的关联关系。"),
        ("sgrq_shap_dependence.png", "SHAP依赖图：SGRQ", "展示生活质量评分升高时，模型风险贡献如何变化。"),
        ("shap_waterfall.png", "SHAP单样本瀑布图", "解释某一位患者为何被预测为高/中/低风险，各特征贡献一目了然。"),
    ]

    for file_name, title, desc in img_map:
        p = Path(file_name)
        st.markdown(f"#### {title}")
        if p.exists():
            safe_show_image(p)
            st.caption(desc)
        else:
            st.warning(f"未找到文件：{file_name}")

    st.markdown("### 其他结果图")
    for p in ["confusion_matrix.png", "roc_curve.png", "xgb_summary.png"]:
        pp = Path(p)
        if pp.exists():
            safe_show_image(pp, caption=p)

# 页面底部锚点（供“向下滑动”按钮定位）
st.markdown('<div id="page-bottom"></div>', unsafe_allow_html=True)

