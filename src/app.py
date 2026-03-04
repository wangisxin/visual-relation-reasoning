"""
Streamlit Web应用 - 多维图像识别与理解
现代化Web界面，支持VLM增强
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="多维图像识别与理解",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入模块
from detector import ObjectDetector
from relation_reasoner import RelationReasoner
from vlm import VLMProcessor
from segmenter import SAMSegmenter
from scene_graph import SceneGraphGenerator

# 初始化检测器 (延迟加载)
@st.cache_resource
def get_detector():
    return ObjectDetector()

@st.cache_resource  
def get_relation_reasoner():
    return RelationReasoner()

@st.cache_resource
def get_vlm():
    return VLMProcessor()

@st.cache_resource
def get_segmenter():
    return SAMSegmenter()

@st.cache_resource
def get_sg_generator():
    return SceneGraphGenerator()


def main():
    # 标题
    st.title("🖼️ 多维图像识别与理解系统")
    st.markdown("**VLM增强版** - 物体检测 · 关系推理 · 场景理解")
    
    # 侧边栏 - 功能选择
    st.sidebar.header("⚙️ 功能选择")
    
    mode = st.sidebar.radio(
        "选择模式",
        ["物体检测", "关系推理", "VLM智能理解", "SAM分割", "场景图生成"],
        captions=[
            "检测图像中的物体",
            "分析物体间空间和语义关系", 
            "AI理解图像内容",
            "一键分割任意物体",
            "生成场景关系图"
        ]
    )
    
    # 侧边栏 - 设置
    st.sidebar.header("🔧 设置")
    confidence = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.5, 0.05)
    show_labels = st.sidebar.checkbox("显示标签", value=True)
    show_scores = st.sidebar.checkbox("显示置信度", value=True)
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "📁 上传图片",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支持JPG, PNG, BMP格式"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # 转换为BGR (OpenCV格式)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
            
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 原图")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("🔍 检测结果")
            
            # 根据模式执行不同功能
            if mode == "物体检测":
                run_detection(image_bgr, confidence, show_labels, show_scores, col2)
            elif mode == "关系推理":
                run_relation_reasoning(image_bgr, confidence, col2)
            elif mode == "VLM智能理解":
                run_vlm(image, col2)
            elif mode == "SAM分割":
                run_sam(image, col2)
            elif mode == "场景图生成":
                run_scene_graph(image_bgr, confidence, col2)
    
    else:
        # 显示示例/欢迎界面
        st.info("👆 请上传一张图片开始分析")
        
        # 显示功能说明
        with st.expander("📖 功能说明", expanded=True):
            st.markdown("""
            ### 🎯 支持的功能
            
            | 功能 | 描述 |
            |------|------|
            | **物体检测** | 识别80类常见物体，显示位置和置信度 |
            | **关系推理** | 分析物体间的空间和语义关系 |
            | **VLM理解** | AI智能描述图像内容，回答问题 |
            | **SAM分割** | 点击物体进行实例分割 |
            | **场景图** | 生成物体-关系-物体的图结构 |
            """)
    
    # 底部信息
    st.markdown("---")
    st.caption(f"🕐 最近更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_detection(image, confidence, show_labels, show_scores, container):
    detector = get_detector()
    results = detector.detect(image, confidence)
    
    # 绘制结果
    annotated = detector.draw_results(image.copy(), results, show_labels, show_scores)
    
    # 显示
    container.image(annotated, channels="BGR", use_container_width=True)
    
    # 显示统计
    if results:
        st.subheader("📊 检测统计")
        stats = detector.get_statistics(results)
        for cls, count in stats.items():
            st.write(f"- {cls}: {count}个")


def run_relation_reasoning(image, confidence, container):
    detector = get_detector()
    reasoner = get_relation_reasoner()
    
    # 检测物体
    detections = detector.detect(image, confidence)
    
    if not detections:
        container.warning("未检测到物体")
        return
    
    # 关系推理
    relations = reasoner.analyze(image, detections)
    
    # 绘制关系
    annotated = reasoner.draw_relations(image.copy(), detections, relations)
    
    container.image(annotated, channels="BGR", use_container_width=True)
    
    # 显示关系
    st.subheader("🔗 关系分析")
    for rel in relations:
        st.write(f"- **{rel['subject']}** {rel['relation']} **{rel['object']}**")


def run_vlm(image, container):
    vlm = get_vlm()
    
    # 生成描述
    description = vlm.describe(image)
    
    container.markdown("### 📝 图像描述")
    container.success(description)
    
    # 问答
    st.markdown("### 💬 问我关于这张图片")
    question = st.text_input("输入问题", key="vlm_question")
    
    if question:
        answer = vlm.answer(image, question)
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")


def run_sam(image, container):
    segmenter = get_segmenter()
    
    st.info("👆 点击图像选择要分割的物体")
    
    # 显示图像 (SAM需要交互)
    container.image(image, use_container_width=True)
    
    # TODO: 实现交互式分割
    container.warning("交互式分割功能开发中...")


def run_scene_graph(image, confidence, container):
    detector = get_detector()
    sg_gen = get_sg_generator()
    
    # 检测
    detections = detector.detect(image, confidence)
    
    if not detections:
        container.warning("未检测到物体")
        return
    
    # 生成场景图
    graph = sg_gen.generate(image, detections)
    
    # 可视化
    annotated = sg_gen.visualize(image, detections, graph)
    
    container.image(annotated, channels="BGR", use_container_width=True)
    
    # 显示图结构
    st.subheader("🕸️ 场景图")
    st.json(graph)


if __name__ == "__main__":
    main()
