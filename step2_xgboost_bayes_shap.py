import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier
import shap


# ==============================
# 0. 全局配置（中文显示 + 警告控制）
# ==============================
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


# ==============================
# 1. 数据读取与准备
# ==============================
data_path = Path("cleaned_data.csv")
if not data_path.exists():
    raise FileNotFoundError("未找到 cleaned_data.csv，请先运行第一步数据清洗脚本。")

df = pd.read_csv(data_path)
target_col = "AECOPD_occur"
if target_col not in df.columns:
    raise ValueError("cleaned_data.csv 中缺少目标列 AECOPD_occur。")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# 7:3 分层划分，保证正负样本分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 处理样本不平衡：scale_pos_weight = 负样本数 / 正样本数
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
if pos_count == 0:
    raise ValueError("训练集正样本数量为0，无法构建二分类模型。")
scale_pos_weight = neg_count / pos_count

print("========== 数据划分信息 ==========")
print(f"总样本量: {len(df)}")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集负样本: {neg_count}, 正样本: {pos_count}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ==============================
# 2. 贝叶斯优化（5折CV，AUC唯一目标）
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def xgb_cv_auc(
    max_depth,
    learning_rate,
    n_estimators,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    reg_alpha,
    reg_lambda,
):
    """
    贝叶斯优化目标函数：
    - 在训练集上做5折分层交叉验证
    - 每折使用验证折作为 early stopping 监控集合
    - 返回5折AUC均值（唯一优化目标）
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(round(max_depth)),
        "learning_rate": float(learning_rate),
        "n_estimators": int(round(n_estimators)),
        "min_child_weight": float(min_child_weight),
        "subsample": float(np.clip(subsample, 0.5, 1.0)),
        "colsample_bytree": float(np.clip(colsample_bytree, 0.5, 1.0)),
        "gamma": float(max(gamma, 0.0)),
        "reg_alpha": float(max(reg_alpha, 0.0)),
        "reg_lambda": float(max(reg_lambda, 0.0)),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 10,
    }

    auc_scores = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 使用 best_iteration 截断预测，充分利用 early stopping 的最佳轮次
        y_val_prob = model.predict_proba(
            X_val, iteration_range=(0, model.best_iteration + 1)
        )[:, 1]
        fold_auc = roc_auc_score(y_val, y_val_prob)
        auc_scores.append(fold_auc)

    return float(np.mean(auc_scores))


# 超参数搜索空间
pbounds = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (80, 500),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 10.0),
}

optimizer = BayesianOptimization(
    f=xgb_cv_auc,
    pbounds=pbounds,
    random_state=42,
    verbose=2,
)

print("\n========== 开始贝叶斯优化 ==========")
optimizer.maximize(init_points=8, n_iter=20)

best_result = optimizer.max
best_params_raw = best_result["params"]

# 处理整数参数，并补齐固定参数
best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": int(round(best_params_raw["max_depth"])),
    "learning_rate": float(best_params_raw["learning_rate"]),
    "n_estimators": int(round(best_params_raw["n_estimators"])),
    "min_child_weight": float(best_params_raw["min_child_weight"]),
    "subsample": float(np.clip(best_params_raw["subsample"], 0.5, 1.0)),
    "colsample_bytree": float(np.clip(best_params_raw["colsample_bytree"], 0.5, 1.0)),
    "gamma": float(max(best_params_raw["gamma"], 0.0)),
    "reg_alpha": float(max(best_params_raw["reg_alpha"], 0.0)),
    "reg_lambda": float(max(best_params_raw["reg_lambda"], 0.0)),
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 10,
}

print("\n========== 贝叶斯优化完成 ==========")
print(f"最优5折CV AUC: {best_result['target']:.6f}")
print("最优参数:")
for k, v in best_params.items():
    print(f"{k}: {v}")


# ==============================
# 3. 使用最优参数训练最终模型（含早停）
# ==============================
final_model = XGBClassifier(**best_params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# 使用最佳迭代轮次进行预测
y_prob = final_model.predict_proba(
    X_test, iteration_range=(0, final_model.best_iteration + 1)
)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)


# ==============================
# 4. 模型评估与可视化
# ==============================
test_auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, zero_division=0)
pre = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n========== 测试集评估结果 ==========")
print(f"AUC      : {test_auc:.6f}")
print(f"Accuracy : {acc:.6f}")
print(f"Recall   : {rec:.6f}")
print(f"Precision: {pre:.6f}")
print(f"F1-score : {f1:.6f}")
print("混淆矩阵:")
print(cm)

# 4.1 混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["预测未发生", "预测发生"],
    yticklabels=["实际未发生", "实际发生"],
)
plt.title("XGBoost 测试集混淆矩阵", fontsize=16)
plt.xlabel("预测类别")
plt.ylabel("实际类别")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 4.2 ROC曲线图
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC曲线 (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="随机基线")
plt.xlabel("假阳性率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("XGBoost 测试集ROC曲线", fontsize=16)
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ==============================
# 5. SHAP解释与图像保存
# ==============================
explainer = shap.TreeExplainer(final_model)
shap_values_raw = explainer.shap_values(X_test)


def _to_shap_matrix(values):
    """兼容不同shap版本输出格式，统一得到二维矩阵 [n_samples, n_features]。"""
    if isinstance(values, list):
        # 老版本二分类有时返回长度2列表，取正类解释值
        if len(values) == 2:
            return np.array(values[1])
        return np.array(values[0])
    if hasattr(values, "values"):
        # 新版可能返回 Explanation 对象
        return np.array(values.values)
    return np.array(values)


shap_matrix = _to_shap_matrix(shap_values_raw)

# 5.1 SHAP汇总图（重要性条形图）
plt.figure()
shap.summary_plot(shap_matrix, X_test, plot_type="bar", show=False)
plt.title("SHAP Summary Plot（重要性条形图）", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.1.1 SHAP beeswarm plot（蜂群图）
plt.figure()
shap.summary_plot(shap_matrix, X_test, show=False)
plt.title("SHAP Beeswarm Plot（蜂群图）", fontsize=14)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()


def choose_feature(col_name, all_cols):
    """若指定列不存在，则尝试用同名前缀列（如PastAE_0/1/2）替代。"""
    if col_name in all_cols:
        return col_name
    prefix_cols = [c for c in all_cols if c.startswith(f"{col_name}_")]
    if prefix_cols:
        # 优先选择编码中的“高等级”列（按名称排序取最后一个）
        prefix_cols = sorted(prefix_cols)
        return prefix_cols[-1]
    return None


# 5.2 核心特征依赖图
dependence_targets = {
    "PastAE": "pastae_shap_dependence.png",
    "FEV1": "fev1_shap_dependence.png",
    "SGRQ": "sgrq_shap_dependence.png",
}

for feat, fig_name in dependence_targets.items():
    real_feat = choose_feature(feat, X_test.columns)
    if real_feat is None:
        print(f"警告：未找到特征 {feat}，跳过 {fig_name} 生成。")
        continue

    plt.figure()
    shap.dependence_plot(
        real_feat,
        shap_matrix,
        X_test,
        interaction_index="auto",
        show=False,
    )
    plt.title(f"SHAP依赖图：{real_feat}", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close()

# 5.3 单样本SHAP瀑布图（随机选取1例）
np.random.seed(42)
sample_idx = np.random.randint(0, X_test.shape[0])

# 为兼容不同shap版本，统一手动构造 Explanation 对象
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
else:
    base_value = explainer.expected_value

single_exp = shap.Explanation(
    values=shap_matrix[sample_idx],
    base_values=base_value,
    data=X_test.iloc[sample_idx].values,
    feature_names=X_test.columns.tolist(),
)

plt.figure()
shap.plots.waterfall(single_exp, show=False, max_display=15)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.4 单样本SHAP force plot（力图）
force_html = shap.force_plot(
    base_value,
    shap_matrix[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=False,
)
shap.save_html("shap_force_plot.html", force_html)

try:
    # shap.force_plot(..., matplotlib=True) 可直接导出为静态PNG
    plt.figure(figsize=(16, 3))
    shap.force_plot(
        base_value,
        shap_matrix[sample_idx],
        X_test.iloc[sample_idx],
        matplotlib=True,
        show=False,
        text_rotation=25,
    )
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
except Exception as exc:
    print(f"警告：force plot PNG 生成失败，仅保留 HTML 版本。原因：{exc}")

# 5.5 SHAP decision plot（决策图）
try:
    plt.figure(figsize=(12, 7))
    shap.decision_plot(
        base_value,
        shap_matrix,
        X_test,
        feature_display_range=slice(-1, -16, -1),
        show=False,
    )
    plt.title("SHAP Decision Plot（决策图）", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_decision_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
except Exception as exc:
    print(f"警告：decision plot 生成失败。原因：{exc}")

# 保存SHAP解释器，便于后续直接加载使用
joblib.dump(explainer, "shap_explainer.pkl")
print("SHAP解释器已保存: shap_explainer.pkl")


# ==============================
# 6. XGBoost特征重要性排序与绘图
# ==============================
importances = final_model.feature_importances_
importance_df = pd.DataFrame(
    {"feature": X_train.columns, "importance": importances}
).sort_values("importance", ascending=False)

print("\n========== XGBoost特征重要性TOP10 ==========")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20),
    x="importance",
    y="feature",
    palette="viridis",
)
plt.title("XGBoost特征重要性（Top 20）", fontsize=16)
plt.xlabel("重要性")
plt.ylabel("特征")
plt.tight_layout()
plt.savefig("xgb_summary.png", dpi=300, bbox_inches="tight")
plt.close()


# ==============================
# 7. 保存模型与评估报告
# ==============================
joblib.dump(final_model, "xgb_model.pkl")
print("\n最优模型已保存: xgb_model.pkl")

report_lines = [
    "========== 模型评估报告 ==========",
    f"最优5折CV AUC: {best_result['target']:.6f}",
    "",
    "最优参数:",
]
for k, v in best_params.items():
    report_lines.append(f"{k}: {v}")

report_lines += [
    "",
    "测试集指标:",
    f"AUC: {test_auc:.6f}",
    f"Accuracy: {acc:.6f}",
    f"Recall: {rec:.6f}",
    f"Precision: {pre:.6f}",
    f"F1-score: {f1:.6f}",
    "",
    "输出文件:",
    "xgb_model.pkl",
    "shap_explainer.pkl",
    "confusion_matrix.png",
    "roc_curve.png",
    "shap_summary.png",
    "shap_beeswarm.png",
    "pastae_shap_dependence.png",
    "fev1_shap_dependence.png",
    "sgrq_shap_dependence.png",
    "shap_waterfall.png",
    "shap_force_plot.html",
    "shap_force_plot.png",
    "shap_decision_plot.png",
    "xgb_summary.png",
]

report_text = "\n".join(report_lines)
with open("xgb_report.txt", "w", encoding="utf-8-sig") as f:
    f.write(report_text)

print("评估报告已保存: xgb_report.txt")
print("\n========== 第二步全部完成 ==========")
