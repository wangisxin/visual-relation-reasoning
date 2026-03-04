# 多维图像识别与理解 - 升级日志

> 记录每一次升级的详细变更

---

## v1.2 (2026-03-04) - VLM/SAM/SceneGraph核心模块

### 日期
2026-03-04

### 阶段
阶段2: VLM融合 + 功能扩展

### 新增功能

#### VLM智能理解模块 (vlm.py)
- **多模式支持**:
  - API模式: OpenAI GPT-4V, 阿里Qwen-VL
  - 本地模型: LLaVA, Qwen-VL支持
  - 轻量模式: 基础图像分析
- 功能:
  - 图像描述生成
  - 智能问答
  - 批量描述

#### SAM分割模块 (segmenter.py)
- **多后端支持**:
  - Meta SAM: vit_b/vit_l/vit_h
  - Grounded-SAM: 检测+分割一体化
  - 轻量模式: OpenCV GrabCut
- 功能:
  - 点/框提示分割
  - 全自动分割 (Segment Everything)
  - 掩码优化

#### 场景图生成模块 (scene_graph.py)
- **多推理模式**:
  - 规则模式: 基于几何位置的关系推理
  - LLM模式: 大语言模型关系推理
  - 神经网络模式: RelTR等方法
- 功能:
  - 空间关系: above/below/left/right/inside/overlaps
  - 语义关系: holding/sitting_on/using
  - 视觉属性: 颜色/大小分析
  - 多格式导出: JSON/Graphviz/Cytoscape

#### 检测器优化 (detector.py)
- 模型尺寸选择: n/s/m/l/x
- 批量推理支持
- 视频流检测
- 性能优化: 颜色缓存/NMS
- 过滤与统计功能

### 文件变更
| 文件 | 操作 | 描述 |
|------|------|------|
| src/vlm.py | 重写 | VLM智能理解 (10000+行) |
| src/segmenter.py | 重写 | SAM分割模块 (11000+行) |
| src/scene_graph.py | 重写 | 场景图生成 (15000+行) |
| src/detector.py | 重写 | YOLOv8优化版 (13000+行) |
| CHANGELOG.md | 新增 | 升级日志 |

### Git提交
`6e25b78` - feat: 阶段2 - VLM/SAM/SceneGraph核心模块实现

### 状态
✅ 已完成
