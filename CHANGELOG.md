# 多维图像识别与理解 - 升级日志

> 记录每一次升级的详细变更

---

## v1.1 (2026-03-04) - Streamlit基础框架

### 日期
2026-03-04

### 阶段
阶段1: 基础架构重构

### 新增功能
- Streamlit Web应用框架
- VLM智能理解模块 (骨架)
- SAM分割模块 (骨架)
- SceneGraph场景图生成模块 (骨架)

### 文件变更
| 文件 | 操作 | 描述 |
|------|------|------|
| src/app.py | 新增 | Streamlit Web界面 |
| src/vlm.py | 新增 | VLM模块 |
| src/segmenter.py | 新增 | SAM分割模块 |
| src/scene_graph.py | 新增 | 场景图模块 |
| requirements.txt | 更新 | 添加streamlit依赖 |
| README.md | 更新 | 技术栈说明 |

### Git提交
`8617249` - feat: 阶段1 - Streamlit基础框架 + VLM模块架构

### 状态
✅ 已完成
