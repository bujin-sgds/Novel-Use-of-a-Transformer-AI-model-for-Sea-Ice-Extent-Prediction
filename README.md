# ZhiQiu Model v8.5 — Transformer-based Sea Ice Prediction

> 一键上手的 **GitHub 开源模板**，已包含 README、许可协议、依赖清单、.gitignore、Git LFS 配置等。将你的代码放进 `src/` 目录即可提交。

## 目录结构

```
.
├─ src/                  # 源代码（已放入 ZhiQiuModel V8.5.py 与无空格版本）
├─ scripts/              # 辅助脚本（可按需扩展）
├─ examples/             # 使用示例与文档草稿
├─ .gitattributes        # Git LFS 追踪大文件（*.pt/*.pkl/*.nc 等）
├─ .gitignore            # 忽略缓存、数据、模型等
├─ LICENSE               # 开源许可证（MIT，按需替换）
├─ CITATION.cff          # 学术引用信息（按需填写作者）
├─ requirements.txt      # Python 依赖（pip）
└─ environment.yml       # Conda 环境（可选）
```

## 快速开始

1. **创建虚拟环境**
   - Conda：`conda env create -f environment.yml && conda activate zhiqiu-env`
   - 或者：`python -m venv .venv && .venv\Scripts\activate`（Win）

2. **安装依赖**
   - `pip install -r requirements.txt`

3. **准备数据**
   - 将 `.nc`、`/arco`、`/weekly` 放在项目根目录（或修改代码中的路径）。
   - 大体积数据与模型 **不要提交到 Git**，已在 `.gitignore` 中忽略；如需版本化，请使用 **Git LFS**。

4. **运行**
   - 进入 `src/`，运行你的主脚本，例如：  
     `python "ZhiQiuModel V8.5.py"`  
     或使用无空格版本：  
     `python ZhiQiuModel_V8_5.py`

## Git LFS（可选但推荐）

项目已配置 `.gitattributes`，自动追踪：`*.pt` `*.pth` `*.pkl` `*.nc` `*.npz` `*.h5`。

- 安装：<https://git-lfs.com>
- 初始化：`git lfs install`

## 引用

请在 `CITATION.cff` 中完善作者信息；GitHub 将自动渲染引用格式。

## 许可

本项目默认使用 MIT 协议，按需替换为你需要的开源许可证。

## 📊 实验结果与性能评估

本模型（**知秋模型 V8.5**）在北极海冰密集度（SIC）预测任务中，针对测试集的综合性能如下：

| 指标 | 模型表现 | 气候态基准 | 持续性基准 |
|------|----------|------------|------------|
| **RMSE** (SIC) | **0.0720** | 0.0657 | 0.0989 |
| **MAE** (SIC) | **0.0184** | - | - |
| **R²** (SIC) | **0.9368** | - | - |
| **SIE RMSE** (百万 km²) | **0.555** | - | - |
| **SIE MAE** (百万 km²) | **0.420** | - | - |

> **说明**：
> - **SIC（Sea Ice Concentration）** 指每个网格单元的海冰浓度。
> - **SIE（Sea Ice Extent）** 是宏观统计量，反映总的海冰覆盖面积。
> - RMSE 越低越好，R² 越高越好。

---

### 📌 综合性能可视化

*左上：像素级预测与真实值对比（R²=0.9368）；左下：空间平均绝对误差（MAE）；右下：空间预测不确定性（MC Dropout Std Dev）*

---

### 📌 SIE 时间序列对比

*黑色为观测值，红色为模型预测，阴影为 95% 置信区间。可以看到模型在 2021–2022 年度的月度变化趋势上与真实观测高度吻合。*
