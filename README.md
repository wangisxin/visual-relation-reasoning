# 图像关系推理系统 VLM增强版

基于 YOLOv8 + DINO-X + SAM 的多维图像识别与理解桌面应用。

## 功能

- [x] 物体检测：识别图片中的 80 类常见物体
- [x] 空间关系推理：判断物体之间的上下、左右、遮挡、前后关系
- [x] 语义关系推理：理解物体之间的语义关联（人拿着、坐在、使用等）
- [x] 物体计数：统计各类别物体数量
- [ ] **VLM智能描述**: 基于DINO-X的图像内容理解
- [ ] **SAM分割**: Segment Anything一键分割
- [ ] **Scene Graph**: 场景图生成与可视化

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# Web界面 (Streamlit)
cd src
streamlit run app.py

# 或命令行模式
python cli.py
```

## 项目结构

```
多维图像识别与理解/
├── docs/plans/          # 计划文档
├── src/
│   ├── detector.py      # 物体检测模块 (YOLOv8)
│   ├── relation_reasoner.py  # 关系推理模块
│   ├── vlm.py           # VLM智能描述模块
│   ├── segmenter.py    # SAM分割模块
│   ├── scene_graph.py  # 场景图生成
│   ├── app.py          # Streamlit Web界面
│   ├── cli.py          # 命令行界面
│   └── main.py         # 桌面UI (Tkinter)
├── requirements.txt    # 依赖
└── README.md
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 物体检测 | YOLOv8, Grounding DINO |
| 分割 | SAM (Segment Anything) |
| VLM | DINO-X |
| 关系推理 | 自研规则 + LLM |
| UI | Streamlit, Tkinter |
