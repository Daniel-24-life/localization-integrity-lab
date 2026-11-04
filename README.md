# Localization Integrity Lab · 移动机器人定位完整性研究

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-116%20passed-brightgreen.svg)]()
[![Portfolio](https://img.shields.io/badge/Showcase-AIfile%20%2314-indigo.svg)](https://github.com/Daniel-24-life/AIfile)

**移动机器人动态定位误差传播、吸收边界 PDE 越界概率与有色噪声自适应滤波可复现研究基准**  
运动学 ODE 建模 · 误差可观测性 Gramian · 吸收边界 PDE (Kolmogorov 首次越界) · EKF 增广状态滤波 · 斯坦福图完好性评测

[English](README.en.md) | [中文说明](README.md)

</div>

---

## 📌 核心研究问题

在移动机器人（AGV / AMR / 自动驾驶小车）的高安全等级自主导航中：
> **轮速打滑与观测退化如何演化为不可观测的定位误差？系统为什么可能已经越界（Hazardously Misleading Information, HMI），却因滤波协方差不匹配而无法触发告警？**

本项目构建了一个从**底层物理运动学 ODE**、**状态误差线性化传播**，到**偏微分方程（PDE）吸收边界首次越界概率计算**与**扩展卡尔曼滤波（EKF）有色噪声状态增广**的完整数学与可复现实验闭环。

---

## 🏛️ 系统研究架构 (Research Architecture)

![Research Architecture](assets/screenshots/research_overview.svg)

系统由五大核心数学与实验阶段构成：
1. **运动学 ODE 与打滑建模**：平面轮式差速运动学模型，显式引入纵向打滑率 $\alpha_v$、旋转打滑率 $\alpha_\omega$、Ornstein-Uhlenbeck (OU) 陀螺漂移以及天线杆臂偏置 $\mathbf{l}^b$；
2. **误差动力学与 Gramian 可观测性**：基于李雅普诺夫/柯西连续线性化矩阵指数 Taylor 展开，计算白化观测 Gramian $\mathbf{W}_{\text{obs}}$ 奇异值与条件数下界；
3. **吸收边界 PDE (首次越界风险)**：
   $$\frac{\partial u}{\partial t} = \mu \frac{\partial u}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2}, \quad u(\pm L, t) = 1$$
   推导纯布朗运动与漂移扩散过程的解析谱展开级数解，为安全告警限（Alert Limit）越界概率提供理论基准，并与 2 万条 Monte Carlo 路径交叉核对；
4. **有色噪声状态增广 EKF 滤波器群**：实现固定白噪声 EKF (B0)、新息协方差匹配自适应 EKF (B3)、有色 OU 状态增广 EKF 以及多模型假设检验滤波库 (Candidate Bank V2)；
5. **斯坦福图完好性评测**：将定位误差（PE）与保护级（PL）、告警限（AL）映射为正常可用、安全告警、系统退化与致命误导（HMI）四大运行象限。

---

## 📊 关键实验发现与反例权衡 (Key Findings & Trade-Offs)

在严谨的 144 条件合成研究矩阵（8 个随机种子 × 2 种运动轨迹 × 3 种噪声环境 × 3 种估计模型）中，实验证实了一个基本科学规律：

| 实验噪声场景 | 滤波器类型 | ATE 均方根误差 (m) | 危险误导 (HMI) 历时比 | 安全可用度历时比 | 误告警历时比 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **匹配有色噪声 (OU Process)** | 固定白噪声 EKF (B0) | 1.371 m | **88.31%** (致命误导) | 10.01% | 0.22% |
| **匹配有色噪声 (OU Process)** | **有色噪声增广 EKF (Proposed)** | **0.135 m** | **0.00% (彻底杜绝 HMI)** | **38.31%** | 61.69% |
| **纯白高斯噪声 (White Only)** | 固定白噪声 EKF (B0) | **0.057 m** | 0.00% | **99.62%** | 0.38% |
| **纯白高斯噪声 (White Only)** | 有色噪声增广 EKF (Proposed) | 0.065 m | 0.00% | 38.28% | 61.72% (过于保守) |

<div align="center">
  <img src="assets/screenshots/synthetic_comparison.png" width="600" alt="Synthetic Benchmark Comparison" />
  <p><em>图：有色噪声模型与白噪声基准在不同工况下的误差与可用度表现</em></p>
</div>

> **科学结论**：状态增广滤波没有“免费午餐”。在存在时间相关性有色噪声时，增广状态能将 HMI 历时比从 88.3% 清零；但在纯白噪声环境下，增广协方差会带来过度保守性，使可用度下降约 61%。因此完好性监测必须引入动态多模型判决。

---

## 🚀 快速上手与 100% 独立复现

无需下载庞大的原始 ROS bag，代码内置确定性合成轨迹发生器，可秒级复现完整 144 次科研评测：

### 1. 安装环境
```bash
git clone https://github.com/Daniel-24-life/localization-integrity-lab.git
cd localization-integrity-lab
pip install -r requirements.txt
```

### 2. 执行 144 条件合成研究复现
```bash
python scripts/run_synthetic_study.py
```
*预期输出：自动生成 8 个种子的运行历程，并在终端输出 9 个维度的汇总指标表与对应 CSV 记录。*

### 3. 运行全套单元测试
```bash
python -m pytest -c pytest.ini tests/ -v
```
*预期输出：116 项物理边界、ODE 一致性、Gramian 满秩与 PDE 级数解析验证测试 100% 通过 (约 13s)。*

---

## 🛡️ 学术严谨性与研究边界声明 (Limitations)

为保持严谨科研作风，本项目如实保留以下边界结论（详见 [`docs/research_limitations.md`](docs/research_limitations.md)）：
1. **样本量归因**：144 次评测是 8 个独立随机种子在不同物理条件下的排列组合，8 个种子属于合成重复实验单位，不可等同于 144 次独立物理实验；
2. **真实场景推广 (G4 门槛)**：在真实室内外复杂打滑与遮挡数据初测中，严格 0.5m 告警限下的真实安全可用度约为 3.26%，**尚未通过原项目 G4 工业部署认证门槛**；
3. **保护级风险标定**：当前 PL 来源于高斯假设下的线性化状态协方差椭球，尚未完成针对极值尾部风险的正式航空/车载级完好性风险分配（$P_{\text{HMI}} < 10^{-7}$）。

---

## 🔗 相关链接与生态
- **数学推导全集**：[`docs/mathematical_formulation.md`](docs/mathematical_formulation.md)
- **AIfile 作品集收录 (#14)**：[https://github.com/Daniel-24-life/AIfile](https://github.com/Daniel-24-life/AIfile)
- **开发者个人主页**：[Daniel-24-life](https://github.com/Daniel-24-life)
