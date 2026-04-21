import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# ==============================
# 0. 基础设置：中文显示与警告处理
# ==============================
# 按顺序尝试常见中文字体，兼容不同系统（Windows/macOS/Linux）
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
# 解决坐标轴负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


# ==============================
# 1. 读取数据并查看基础信息
# ==============================
input_path = "dataset.csv"
output_clean_path = "cleaned_data.csv"
output_heatmap_path = "corr_heatmap.png"

df = pd.read_csv(input_path)

print("========== 原始数据基本信息 ==========")
print(f"数据行数: {df.shape[0]}")
print(f"数据列数: {df.shape[1]}")
print("\n字段类型:")
print(df.dtypes)
print("\n前5行样例:")
print(df.head())

# 删除无效列（如 ID）
invalid_cols = ["ID", "id", "Id"]
drop_cols = [c for c in invalid_cols if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"\n已删除无效列: {drop_cols}")
else:
    print("\n未检测到需删除的无效列（ID/id/Id）")


# ==============================
# 2. 缺失值处理
#    - 数值变量: 中位数填充
#    - 分类变量: 众数填充
# ==============================
# 根据题目定义显式指定分类变量和连续变量，保证处理逻辑清晰可控
categorical_cols = [
    "PastAE",
    "Smoke",
    "Infect",
    "AirPollu",
    "Cardio",
    "Diabetes",
    "Bronchiect",
    "TempChange",
    "Psych",
    "FollowUp",
    "AECOPD_occur",
]
continuous_cols = ["FEV1", "Age", "BMI", "Fibrinogen", "SGRQ"]

# 仅处理实际存在于数据中的列（增强脚本鲁棒性）
categorical_cols = [c for c in categorical_cols if c in df.columns]
continuous_cols = [c for c in continuous_cols if c in df.columns]

print("\n========== 缺失值统计（处理前） ==========")
print(df.isnull().sum())

if continuous_cols:
    num_imputer = SimpleImputer(strategy="median")
    df[continuous_cols] = num_imputer.fit_transform(df[continuous_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\n========== 缺失值统计（处理后） ==========")
print(df.isnull().sum())


# ==============================
# 3. 异常值处理（IQR法）
#    对连续变量进行“修正”（winsorize到上下限）
# ==============================
outlier_summary = {}

for col in continuous_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    lower_count = int((df[col] < lower).sum())
    upper_count = int((df[col] > upper).sum())
    outlier_summary[col] = {
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_outliers": lower_count,
        "upper_outliers": upper_count,
    }

    # 使用clip将异常值修正到边界值，而不是直接删除样本
    df[col] = df[col].clip(lower=lower, upper=upper)

print("\n========== 连续变量异常值处理结果（IQR） ==========")
for col, info in outlier_summary.items():
    print(
        f"{col}: 下界={info['lower_bound']:.3f}, 上界={info['upper_bound']:.3f}, "
        f"下界异常={info['lower_outliers']}, 上界异常={info['upper_outliers']}"
    )


# ==============================
# 4. 多分类变量独热编码（One-Hot，1/0）
# ==============================
multi_class_cols = [c for c in ["PastAE", "Smoke", "AirPollu", "Psych"] if c in df.columns]

df_encoded = pd.get_dummies(
    df,
    columns=multi_class_cols,
    prefix=multi_class_cols,
    prefix_sep="_",
    dtype=int,
)

print("\n========== 编码后数据维度 ==========")
print(f"编码后行数: {df_encoded.shape[0]}")
print(f"编码后列数: {df_encoded.shape[1]}")


# ==============================
# 5. 相关性分析与热力图保存
# ==============================
corr_matrix = df_encoded.corr(numeric_only=True)

# 只显示下三角，减少信息拥挤，提升文字可读性
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(32, 28))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    square=False,
    linewidths=0.3,
    annot=True,           # 在每个格子中显示相关系数
    fmt=".2f",            # 保留2位小数
    annot_kws={"size": 16},  # 大幅放大系数字体
    cbar_kws={"shrink": 0.8, "label": "相关系数"},
)
plt.title("清洗与编码后特征相关性热力图（下三角）", fontsize=34, pad=24)
plt.xticks(rotation=45, ha="right", fontsize=20)
plt.yticks(rotation=0, fontsize=20)
plt.tight_layout()
plt.savefig(output_heatmap_path, dpi=500, bbox_inches="tight")
plt.close()

print(f"\n相关性热力图已保存: {output_heatmap_path}")


# ==============================
# 6. 保存清洗后的数据
# ==============================
df_encoded.to_csv(output_clean_path, index=False, encoding="utf-8-sig")
print(f"清洗后的数据已保存: {output_clean_path}")

print("\n========== 全流程完成 ==========")
