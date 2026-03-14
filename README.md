<p align="center">
  <img src="docs/logo.png" alt="ClimKit logo" width="200" />
</p>
# 📘 ClimKit — A Climate Data Analysis Toolkit

**ClimKit** 是一个用于气候数据及绘图的 Python 工具集。它包含多种常用数据处理工具，例如波谱分析、温度收支、矢量场可视化等，适用于科研人员、地球科学研究者及数据分析工程师。

---

## 🌟 功能概览

ClimKit 提供以下核心功能模块：

### 🔸 1. **Wavelet Analysis（小波分析和功率谱分析）**

位于 `climkit.wavelet` 和 `climkit.specx_anal`

* 执行连续小波变换（CWT）
* 分析气候/海洋数据的周期性与多尺度特征
* 提供功率谱、显著性检验等方法

---

### 🔸 2. **Temperature Budget（温度收支分析）**

位于 `climkit.temperature_budget`

* 计算局地温度场的收支项
* 用于大气热力诊断

---

### 🔸 3. **K-Means 聚类工具**

位于 `climkit.K_Mean`

* 常用于气候系统聚类分型分析

---

### 🔸 4. **Cquiver — 矢量场绘图**

位于 `climkit.Cquiver`

* 扩展版 matplotlib.quiver
* 更美观的可视化风场、流场

---

### 🔸 5. **Subfig Adjustment（子图调整）**

位于 `climkit.sub_adjust`

* 绘制地图子图（如：中国黄海；中国南海）

---

### 🔸 6. **Filter （信号滤波）**

位于 `climkit.filter`

* 洛伦兹滤波、滑动滤波、巴特霍夫滤波等
* 用于对信号进行滤波，并可以绘制响应函数

---

### 🔸 7. **Lonlat transform （经度格式转换）**

位于 `climkit.lonlat_transform`

* 对360度和180度制的经度格式提供互相转换功能

---

### 🔸 8. **Masked （数据裁切）**

位于 `climkit.masked`

* 基于SHP文件对NetCDF数据进行数据裁切

---

### 🔸 9. **LBM forcing data （制作线性斜压模式强迫数据）**

位于 `climkit.force_file`

* 制作NetCDF格式的LBM强迫数据
* 目前仅支持T42L20配置下的LBM

---

### 🔸 10. **T-N Wave Activity Flux （计算T-N波活动通量）**

位于 `climkit.TN_WaveActivityFlux`

* 基于气候态水平风速场和异常位势高度场计算波活动通量
* 支持二维和三维波活动通量的计算（三维波活动通量需要提供气候态温度场）

---

## 🚀 安装方式

### **方式一：pip（推荐）**

```bash
pip install climkit
```

### **方式二：从源码安装**

```bash
git clone https://example.com/climkit.git
cd climkit
pip install -e .
```

---

## 🧪 快速开始

---

### 示例：矢量风场绘图

```python
from climkit.Cquiver import Curlyquiver
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection=*)
ax.Curlyquiver(x, y, U, V)
plt.show()
```

---




