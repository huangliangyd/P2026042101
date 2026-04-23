import base64
from pathlib import Path

import joblib
import m2cgen as m2c
import pandas as pd


ROOT = Path(".")
MODEL_PATH = ROOT / "xgb_model.pkl"
DATA_PATH = ROOT / "cleaned_data.csv"
REPORT_PATH = ROOT / "xgb_report.txt"
OUT_HTML = ROOT / "aecopd_offline_interactive.html"


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_report(path: Path):
    params, metrics = {}, {}
    if not path.exists():
        return params, metrics
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    in_params, in_metrics = False, False
    for line in lines:
        line = line.strip()
        if line == "最优参数:":
            in_params, in_metrics = True, False
            continue
        if line == "测试集指标:":
            in_params, in_metrics = False, True
            continue
        if line == "输出文件:" or line.startswith("输出文件:"):
            in_params, in_metrics = False, False
            continue
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if in_params:
            params[k.strip()] = v.strip()
        elif in_metrics:
            metrics[k.strip()] = v.strip()
    return params, metrics


def format_decimal_str(value: str, decimals: int = 2) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if num.is_integer():
        return str(int(num))
    return f"{num:.{decimals}f}"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("缺少 xgb_model.pkl")
    if not DATA_PATH.exists():
        raise FileNotFoundError("缺少 cleaned_data.csv")

    model = joblib.load(MODEL_PATH)
    # 兼容部分xgboost版本：base_score可能为None，m2cgen导出时会报错
    if getattr(model, "base_score", None) is None:
        model.base_score = 0.5
    df = pd.read_csv(DATA_PATH)
    feature_names = list(getattr(model, "feature_names_in_", [c for c in df.columns if c != "AECOPD_occur"]))

    # 导出 JS 评分函数：返回 logit，需要再做 sigmoid 转概率
    js_expr = m2c.export_to_javascript(model)
    params, metrics = parse_report(REPORT_PATH)

    images = {
        "corr": img_to_base64(ROOT / "corr_heatmap.png"),
        "cm": img_to_base64(ROOT / "confusion_matrix.png"),
        "roc": img_to_base64(ROOT / "roc_curve.png"),
        "xgbsum": img_to_base64(ROOT / "xgb_summary.png"),
        "shap_summary": img_to_base64(ROOT / "shap_summary.png"),
        "shap_pastae": img_to_base64(ROOT / "pastae_shap_dependence.png"),
        "shap_fev1": img_to_base64(ROOT / "fev1_shap_dependence.png"),
        "shap_sgrq": img_to_base64(ROOT / "sgrq_shap_dependence.png"),
        "shap_waterfall": img_to_base64(ROOT / "shap_waterfall.png"),
    }

    params_html = "".join(
        [f"<li><b>{k}</b>: {format_decimal_str(v, 2)}</li>" for k, v in params.items()]
    )
    metric_order = ["AUC", "Accuracy", "Recall", "Precision", "F1-score"]
    metrics_html = "".join(
        [
            f"<div class='metric'><span>{k}</span><strong>{format_decimal_str(metrics[k], 4)}</strong></div>"
            for k in metric_order
            if k in metrics
        ]
    )
    features_js = "[" + ",".join([f"'{f}'" for f in feature_names]) + "]"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AECOPD风险智能预测平台（离线版）</title>
  <style>
    body {{ margin:0; font-family: "Microsoft YaHei", Arial, sans-serif; background:#f5f8fc; color:#1b2a3a; }}
    .layout {{ display:flex; min-height:100vh; }}
    .sidebar {{ width:240px; background:#0b2c59; color:#fff; padding:20px 16px; }}
    .hospital {{ font-size:24px; font-weight:700; margin-bottom:24px; }}
    .navbtn {{ width:100%; padding:12px; margin-bottom:10px; border:none; border-radius:8px; background:#174a8b; color:#fff; cursor:pointer; font-size:16px; }}
    .navbtn.active {{ background:#2b6cb0; }}
    .content {{ flex:1; padding:24px; }}
    .page {{ display:none; }}
    .page.active {{ display:block; }}
    h1 {{ color:#0b2c59; border-left:6px solid #0b2c59; padding-left:10px; }}
    .card {{ background:#fff; border-radius:12px; padding:16px; margin-bottom:14px; box-shadow:0 2px 8px rgba(11,44,89,.08); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
    label {{ display:block; font-weight:600; margin-bottom:6px; }}
    input, select {{ width:100%; padding:8px; border:1px solid #cdd6e1; border-radius:8px; }}
    .predict-btn {{ margin-top:12px; background:#0b2c59; color:#fff; border:none; padding:12px 18px; border-radius:10px; cursor:pointer; font-size:16px; }}
    .result {{ display:flex; gap:10px; margin-top:12px; }}
    .metric {{ background:#edf3fb; border-radius:10px; padding:10px 12px; margin:4px; display:inline-block; min-width:160px; }}
    .metric span {{ display:block; font-size:13px; color:#425466; }}
    .metric strong {{ font-size:20px; color:#0b2c59; }}
    .metric.threshold strong {{ font-size:14px; font-weight:600; line-height:1.45; }}
    img {{ max-width:100%; border-radius:8px; margin-top:8px; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; }}
    th,td {{ border:1px solid #d8e1ee; padding:8px; text-align:left; font-size:14px; }}
    th {{ background:#eef4fb; }}
  </style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="hospital">某某医院</div>
    <button class="navbtn active" onclick="showPage('p1', this)">预测平台</button>
    <button class="navbtn" onclick="showPage('p2', this)">数据探索</button>
    <button class="navbtn" onclick="showPage('p3', this)">模型信息</button>
  </aside>
  <main class="content">
    <section id="p1" class="page active">
      <h1>AECOPD风险智能预测平台</h1>
      <div class="card">
        <p style="margin-top:0;color:#425466;">
          提示：当前模型主要由 <b>PastAE、FEV1、SGRQ</b> 驱动，其他特征在本模型中的贡献较小，数值可能变化不明显。
        </p>
        <div class="grid">
          <div><label>PastAE</label><select id="PastAE"><option>0</option><option>1</option><option>2</option></select></div>
          <div><label>Smoke</label><select id="Smoke"><option>0</option><option>1</option><option>2</option></select></div>
          <div><label>FEV1 (30-80)</label><input id="FEV1" type="number" min="30" max="80" step="0.1" value="55"></div>
          <div><label>Infect</label><select id="Infect"><option>0</option><option>1</option></select></div>
          <div><label>AirPollu</label><select id="AirPollu"><option>0</option><option>1</option><option>2</option></select></div>
          <div><label>Age (50-85)</label><input id="Age" type="number" min="50" max="85" step="1" value="65"></div>
          <div><label>BMI (16-30)</label><input id="BMI" type="number" min="16" max="30" step="0.1" value="22"></div>
          <div><label>Cardio</label><select id="Cardio"><option>0</option><option>1</option></select></div>
          <div><label>Diabetes</label><select id="Diabetes"><option>0</option><option>1</option></select></div>
          <div><label>Bronchiect</label><select id="Bronchiect"><option>0</option><option>1</option></select></div>
          <div><label>TempChange</label><select id="TempChange"><option>0</option><option>1</option></select></div>
          <div><label>Psych</label><select id="Psych"><option>0</option><option>1</option><option>2</option></select></div>
          <div><label>Fibrinogen (2.0-5.0)</label><input id="Fibrinogen" type="number" min="2" max="5" step="0.1" value="3.2"></div>
          <div><label>SGRQ (10-90)</label><input id="SGRQ" type="number" min="10" max="90" step="1" value="45"></div>
          <div><label>FollowUp</label><select id="FollowUp"><option>0</option><option>1</option></select></div>
        </div>
        <button class="predict-btn" onclick="predictRisk()">一键预测</button>
        <div class="result">
          <div class="metric"><span>AECOPD发生风险概率</span><strong id="riskProb">-</strong></div>
          <div class="metric"><span>风险等级</span><strong id="riskLevel">-</strong></div>
          <div class="metric threshold"><span>判定阈值</span><strong>低风险：&lt;0.3　|　中风险：0.3–0.6　|　高风险：&gt;0.6</strong></div>
        </div>
      </div>
      <div class="card">
        <h3>SHAP瀑布图</h3>
        <p style="color:#425466;">离线HTML中展示为参考图；若需“当前患者实时SHAP瀑布图”，请使用 Streamlit 版本运行。</p>
        <img src="data:image/png;base64,{images["shap_waterfall"]}" />
      </div>
    </section>
    <section id="p2" class="page">
      <h1>数据说明</h1>
      <div class="card">
        <p><b>数据集概况：</b>样本量 {len(df)}，研究目标为预测AECOPD_occur（0=未发生，1=发生），数据来源于某某医院临床科研队列（脱敏）。</p>
        <p><b>数据预处理：</b>连续变量中位数填补、分类变量众数填补；IQR法处理异常值；多分类采用独热编码（1/0）。</p>
      </div>
      <div class="card">
        <h3>输入特征释义</h3>
        <table>
          <tr><th>变量</th><th>临床含义</th><th>取值</th><th>编码说明</th></tr>
          <tr><td>PastAE</td><td>既往AECOPD病史</td><td>0/1/2</td><td>0=无，1=1次，2=≥2次</td></tr>
          <tr><td>Smoke</td><td>吸烟状态</td><td>0/1/2</td><td>0=不吸烟，1=既往，2=现吸烟</td></tr>
          <tr><td>FEV1</td><td>肺功能FEV1%预测值</td><td>30~80</td><td>连续</td></tr>
          <tr><td>Infect</td><td>近期感染</td><td>0/1</td><td>0=无，1=有</td></tr>
          <tr><td>AirPollu</td><td>空气污染程度</td><td>0/1/2</td><td>0低 1中 2高</td></tr>
          <tr><td>Age</td><td>年龄</td><td>50~85</td><td>连续</td></tr>
          <tr><td>BMI</td><td>体质量指数</td><td>16~30</td><td>连续</td></tr>
          <tr><td>Cardio</td><td>合并心血管疾病</td><td>0/1</td><td>0=无，1=有</td></tr>
          <tr><td>Diabetes</td><td>合并糖尿病</td><td>0/1</td><td>0=无，1=有</td></tr>
          <tr><td>Bronchiect</td><td>合并支气管扩张</td><td>0/1</td><td>0=无，1=有</td></tr>
          <tr><td>TempChange</td><td>温度变化暴露</td><td>0/1</td><td>0=少，1=多</td></tr>
          <tr><td>Psych</td><td>负性心理状态</td><td>0/1/2</td><td>0=无，1=轻度，2=中重度</td></tr>
          <tr><td>Fibrinogen</td><td>纤维蛋白原(g/L)</td><td>2.0~5.0</td><td>连续</td></tr>
          <tr><td>SGRQ</td><td>呼吸问卷评分</td><td>10~90</td><td>分值越高健康越差</td></tr>
          <tr><td>FollowUp</td><td>复诊规律</td><td>0/1</td><td>0规律，1不规律</td></tr>
        </table>
      </div>
      <div class="card"><h3>特征相关性热力图</h3><img src="data:image/png;base64,{images["corr"]}" /></div>
    </section>
    <section id="p3" class="page">
      <h1>模型说明</h1>
      <div class="card">
        <h3>模型构建流程</h3>
        <p>XGBoost二分类模型，目标函数binary:logistic，贝叶斯优化+5折交叉验证调参，AUC为唯一优化目标。</p>
        <h3>最优参数</h3>
        <ul>{params_html}</ul>
        <h3>模型性能指标</h3>
        <div>{metrics_html}</div>
      </div>
      <div class="card"><h3>SHAP Summary Plot（重要性条形图）</h3><img src="data:image/png;base64,{images["shap_summary"]}" /><p>按平均绝对SHAP值排序展示全局特征重要性。</p></div>
      <div class="card"><h3>SHAP依赖图：PastAE / FEV1 / SGRQ</h3>
        <img src="data:image/png;base64,{images["shap_pastae"]}" /><p>PastAE病史水平变化对应的风险贡献趋势。</p>
        <img src="data:image/png;base64,{images["shap_fev1"]}" /><p>FEV1变化与模型风险贡献关系。</p>
        <img src="data:image/png;base64,{images["shap_sgrq"]}" /><p>SGRQ评分变化对预测风险的影响趋势。</p>
      </div>
      <div class="card"><h3>单样本SHAP瀑布图</h3><img src="data:image/png;base64,{images["shap_waterfall"]}" /><p>个体层面风险归因展示。</p></div>
      <div class="card"><h3>补充评估图</h3>
        <img src="data:image/png;base64,{images["cm"]}" />
        <img src="data:image/png;base64,{images["roc"]}" />
        <img src="data:image/png;base64,{images["xgbsum"]}" />
      </div>
    </section>
  </main>
</div>
<script>
const modelFeatures = {features_js};
function showPage(id, btn){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.navbtn').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}

{js_expr}

function buildVector() {{
  const raw = {{
    PastAE: parseInt(document.getElementById('PastAE').value),
    Smoke: parseInt(document.getElementById('Smoke').value),
    FEV1: parseFloat(document.getElementById('FEV1').value),
    Infect: parseInt(document.getElementById('Infect').value),
    AirPollu: parseInt(document.getElementById('AirPollu').value),
    Age: parseFloat(document.getElementById('Age').value),
    BMI: parseFloat(document.getElementById('BMI').value),
    Cardio: parseInt(document.getElementById('Cardio').value),
    Diabetes: parseInt(document.getElementById('Diabetes').value),
    Bronchiect: parseInt(document.getElementById('Bronchiect').value),
    TempChange: parseInt(document.getElementById('TempChange').value),
    Psych: parseInt(document.getElementById('Psych').value),
    Fibrinogen: parseFloat(document.getElementById('Fibrinogen').value),
    SGRQ: parseFloat(document.getElementById('SGRQ').value),
    FollowUp: parseInt(document.getElementById('FollowUp').value)
  }};
  let feat = {{}};
  modelFeatures.forEach(f => feat[f] = 0.0);
  ['FEV1','Infect','Age','BMI','Cardio','Diabetes','Bronchiect','TempChange','Fibrinogen','SGRQ','FollowUp']
    .forEach(k => {{ if (k in feat) feat[k] = raw[k]; }});
  ['PastAE','Smoke','AirPollu','Psych'].forEach(k => {{
    const col = `${{k}}_${{raw[k]}}`;
    if (col in feat) feat[col] = 1.0;
  }});
  return modelFeatures.map(f => feat[f]);
}}

function predictRisk() {{
  const vec = buildVector();
  const output = score(vec);
  // m2cgen 对二分类模型通常返回 [p0, p1]；若返回单值则按logit转概率
  const prob = Array.isArray(output)
    ? Number(output[output.length - 1])
    : (1 / (1 + Math.exp(-Number(output))));
  let lvl = '高风险';
  if (prob < 0.3) lvl = '低风险';
  else if (prob <= 0.6) lvl = '中风险';
  document.getElementById('riskProb').innerText = prob.toFixed(4);
  document.getElementById('riskLevel').innerText = lvl;
}}

</script>
</body>
</html>
"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"离线交互HTML已生成：{OUT_HTML}")


if __name__ == "__main__":
    main()

