"""
Streamlit Web应用 - 多维图像识别与理解 V2.0
现代化Web界面，支持VLM增强，性能优化
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

# ================== 页面配置 ==================
st.set_page_config(
    page_title="🖼️ 多维图像识别与理解",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''
        ## 多维图像识别与理解系统 V2.0
        
        基于 YOLOv8 + VLM + SAM 的智能图像理解系统
        
        - 物体检测: YOLOv8
        - 智能理解: DINO-X / GPT-4V
        - 实例分割: SAM
        - 场景图: 自研算法
        '''
    }
)

# ================== 自定义CSS ==================
st.markdown("""
<style>
    /* 主色调 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    
    /* 卡片样式 */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* 成功提示 */
    .success-box {
        background: #d1fae5;
        border: 1px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 进度条 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* 侧边栏 */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ================== 模块导入 ==================
try:
    from detector import ObjectDetector
    from relation_reasoner import RelationReasoner
    from vlm import VLMProcessor, create_vlm_processor
    from segmenter import SAMSegmenter, create_sam_segmenter
    from scene_graph import SceneGraphGenerator
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.warning(f"部分模块导入失败: {e}")

# ================== 缓存函数 ==================
@st.cache_resource
def get_detector(model_size: str = 'n', device: str = 'cpu'):
    """获取检测器实例"""
    return ObjectDetector(model_size=model_size, device=device)

@st.cache_resource  
def get_relation_reasoner():
    """获取关系推理器实例"""
    return RelationReasoner()

@st.cache_resource
def get_vlm(config: Optional[Dict] = None):
    """获取VLM处理器"""
    return create_vlm_processor(config)

@st.cache_resource
def get_segmenter(config: Optional[Dict] = None):
    """获取SAM分割器"""
    return create_sam_segmenter(config)

@st.cache_resource
def get_sg_generator(config: Optional[Dict] = None):
    """获取场景图生成器"""
    return SceneGraphGenerator(config)


# ================== 辅助函数 ==================
def load_image(uploaded_file) -> np.ndarray:
    """加载图像"""
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # 转换为BGR (OpenCV格式)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    return image_bgr, image_np

def save_results(image: np.ndarray, detections: List[Dict], 
                 output_path: str) -> str:
    """保存结果图像"""
    cv2.imwrite(output_path, image)
    return output_path

def download_json(data: Dict, filename: str = "results.json"):
    """下载JSON文件"""
    return json.dumps(data, indent=2, ensure_ascii=False)


# ================== 主界面 ==================
def main():
    # 标题
    st.title("🖼️ 多维图像识别与理解系统")
    st.markdown("**VLM增强版 V2.0** | 物体检测 · 关系推理 · 场景理解")
    
    # ================== 侧边栏 ==================
    st.sidebar.header("⚙️ 功能配置")
    
    # 功能选择
    mode = st.sidebar.radio(
        "🎯 选择功能",
        ["物体检测", "关系推理", "VLM智能理解", "SAM分割", "场景图生成", "一键分析"],
        captions=[
            "识别图像中的物体",
            "分析物体间空间和语义关系", 
            "AI智能理解图像内容",
            "一键分割任意物体",
            "生成物体关系图",
            "完整分析所有功能"
        ]
    )
    
    # 检测设置
    st.sidebar.header("🔧 检测设置")
    confidence = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.5, 0.05, key="conf_slider")
    iou_threshold = st.sidebar.slider("NMS IoU阈值", 0.1, 1.0, 0.45, 0.05)
    model_size = st.sidebar.selectbox(
        "模型大小",
        ["n (最快)", "s (快速)", "m (平衡)", "l (精确)", "x (最精)"],
        index=0
    )
    model_size = model_size[0]  # 提取字母
    
    # 显示设置
    st.sidebar.header("🎨 显示设置")
    show_labels = st.sidebar.checkbox("显示标签", value=True)
    show_scores = st.sidebar.checkbox("显示置信度", value=True)
    show_fps = st.sidebar.checkbox("显示处理时间", value=True)
    
    # 高级设置
    with st.sidebar.expander("⚡ 高级设置"):
        enable_gpu = st.checkbox("启用GPU加速", value=False)
        batch_mode = st.checkbox("批量处理模式", value=False)
    
    # ================== 主内容区 ==================
    
    # 文件上传区域
    st.header("📁 上传图片")
    
    col_upload1, col_upload2 = st.columns([3, 1])
    
    with col_upload1:
        uploaded_file = st.file_uploader(
            "支持 JPG, PNG, BMP, WEBP 格式",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="拖拽或点击上传图片",
            label_visibility="collapsed"
        )
    
    with col_upload2:
        use_camera = st.button("📷 使用摄像头", use_container_width=True)
    
    if uploaded_file is not None:
        # 加载图像
        with st.spinner("🔄 加载图像..."):
            image_bgr, image_rgb = load_image(uploaded_file)
        
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["🔍 分析结果", "📊 详细数据", "💾 导出"])
        
        with tab1:
            # 分析图像
            if not MODULES_AVAILABLE:
                st.error("❌ 模块未正确加载，请检查依赖安装")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 根据模式执行
            if mode == "物体检测":
                status_text.text("🔬 进行物体检测...")
                progress_bar.progress(25)
                run_detection(image_bgr, confidence, show_labels, show_scores, show_fps)
                progress_bar.progress(100)
                
            elif mode == "关系推理":
                status_text.text("🔬 物体检测 + 关系推理...")
                progress_bar.progress(25)
                detections = run_detection(image_bgr, confidence, show_labels, show_scores, show_fps, return_results=True)
                progress_bar.progress(50)
                if detections:
                    run_relation_reasoning(image_bgr, detections)
                progress_bar.progress(100)
                
            elif mode == "VLM智能理解":
                status_text.text("🤖 VLM智能理解...")
                progress_bar.progress(25)
                run_vlm(image_rgb)
                progress_bar.progress(100)
                
            elif mode == "SAM分割":
                status_text.text("✂️ SAM分割...")
                progress_bar.progress(25)
                run_sam(image_rgb)
                progress_bar.progress(100)
                
            elif mode == "场景图生成":
                status_text.text("🕸️ 生成场景图...")
                progress_bar.progress(25)
                detections = run_detection(image_bgr, confidence, show_labels, show_scores, show_fps, return_results=True)
                progress_bar.progress(50)
                if detections:
                    run_scene_graph(image_bgr, detections)
                progress_bar.progress(100)
                
            elif mode == "一键分析":
                status_text.text("🚀 一键完整分析...")
                progress_bar.progress(20)
                
                # 物体检测
                detections = run_detection(image_bgr, confidence, show_labels, show_scores, show_fps, return_results=True)
                progress_bar.progress(40)
                
                # 关系推理
                if detections:
                    run_relation_reasoning(image_bgr, detections)
                progress_bar.progress(60)
                
                # 场景图
                if detections:
                    run_scene_graph(image_bgr, detections)
                progress_bar.progress(80)
                
                # VLM描述
                run_vlm(image_rgb)
                progress_bar.progress(100)
            
            status_text.text("✅ 分析完成!")
            progress_bar.empty()
        
        with tab2:
            # 详细数据
            st.subheader("📊 详细数据")
            st.info("此处显示检测统计、关系详情等数据")
            
        with tab3:
            # 导出
            st.subheader("💾 导出结果")
            st.success("分析结果可导出为JSON格式")
    
    else:
        # 欢迎界面
        show_welcome()


def show_welcome():
    """显示欢迎界面"""
    st.info("👆 请上传一张图片开始分析")
    
    # 功能卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 物体检测</h4>
            <p>基于YOLOv8识别80类常见物体</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔗 关系推理</h4>
            <p>分析物体间空间和语义关系</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 VLM理解</h4>
            <p>AI智能描述图像内容</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        ### 快速开始
        
        1. **上传图片**: 点击上方文件选择器或拖拽图片
        2. **选择功能**: 在左侧边栏选择要使用的功能
        3. **调整参数**: 根据需要调整置信度等参数
        4. **查看结果**: 在主区域查看分析结果
        
        ### 功能说明
        
        | 功能 | 描述 |
        |------|------|
        | 物体检测 | 识别图像中的物体并显示位置 |
        | 关系推理 | 分析物体之间的空间和语义关系 |
        | VLM理解 | AI智能描述图像内容并回答问题 |
        | SAM分割 | 使用Segment Anything分割物体 |
        | 场景图 | 生成物体-关系-物体的图结构 |
        | 一键分析 | 完整分析所有功能 |
        
        ### 高级选项
        
        - **模型大小**: n=最快, x=最准
        - **GPU加速**: 需要CUDA支持
        - **批量处理**: 支持多图同时处理
        """)
    
    # 底部信息
    st.markdown("---")
    st.caption(f"🕐 系统版本: V2.0 | 最近更新: {datetime.now().strftime('%Y-%m-%d')}")


# ================== 功能函数 ==================
def run_detection(image, confidence, show_labels, show_scores, show_fps, return_results=False):
    """物体检测"""
    import time
    start_time = time.time()
    
    detector = get_detector(model_size='n')
    results = detector.detect(image, conf_threshold=confidence)
    
    # 绘制结果
    annotated = detector.draw_results(image.copy(), results, show_labels, show_scores)
    
    # 显示图像
    st.image(annotated, channels="BGR", use_container_width=True, caption="检测结果")
    
    # 显示处理时间
    if show_fps:
        elapsed = time.time() - start_time
        st.success(f"⏱️ 处理时间: {elapsed:.2f}秒 | 检测到 {len(results)} 个物体")
    
    # 显示统计
    if results:
        st.subheader("📊 检测统计")
        stats = detector.get_statistics(results)
        
        # 显示为指标卡片
        cols = st.columns(min(len(stats), 4))
        for i, (cls, count) in enumerate(stats.items()):
            cols[i % 4].metric(cls, count)
        
        # 显示详细列表
        with st.expander("📋 详细列表"):
            for det in results:
                st.write(f"- **{det['class_name']}**: {det['confidence']:.2f}")
    
    if return_results:
        return results


def run_relation_reasoning(image, detections):
    """关系推理"""
    reasoner = get_relation_reasoner()
    
    # 关系推理
    relations = reasoner.analyze(image, detections)
    
    # 绘制关系
    annotated = reasoner.draw_relations(image.copy(), detections, relations)
    
    st.image(annotated, channels="BGR", use_container_width=True, caption="关系推理结果")
    
    # 显示关系
    if relations:
        st.subheader("🔗 关系分析")
        
        # 分类显示
        spatial = [r for r in relations if r.get('type') == 'spatial']
        semantic = [r for r in relations if r.get('type') == 'semantic']
        
        if spatial:
            st.markdown("**🧭 空间关系:**")
            for rel in spatial:
                st.write(f"- {rel['subject']} {rel['relation']} {rel['object']}")
        
        if semantic:
            st.markdown("**💡 语义关系:**")
            for rel in semantic:
                st.write(f"- {rel['subject']} {rel['relation']} {rel['object']}")


def run_vlm(image):
    """VLM智能理解"""
    vlm = get_vlm()
    
    # 生成描述
    with st.spinner("🤖 AI正在理解图像..."):
        description = vlm.describe(image)
    
    st.subheader("📝 图像描述")
    st.success(description)
    
    # 问答
    st.subheader("💬 智能问答")
    
    # 预设问题
    preset_questions = [
        "这张图片有什么?",
        "主要场景是什么?",
        "有多少个人?",
        "在做什么事情?"
    ]
    
    cols = st.columns(4)
    for i, q in enumerate(preset_questions):
        if cols[i].button(q, key=f"preset_{i}"):
            answer = vlm.answer(image, q)
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {answer}")
    
    # 自定义问题
    question = st.text_input("自定义问题", placeholder="问我任何关于这张图片的问题...")
    
    if question:
        with st.spinner("🤔 思考中..."):
            answer = vlm.answer(image, question)
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")


def run_sam(image):
    """SAM分割"""
    segmenter = get_segmenter()
    
    # 全自动分割
    with st.spinner("✂️ 正在分割..."):
        results = segmenter.segment_everything(image)
    
    if results:
        st.success(f"分割完成! 发现 {len(results)} 个物体")
        
        # 显示分割结果数量
        st.metric("分割物体数", len(results))
        
        # TODO: 实现交互式选择
        st.info("💡 点击物体进行精确分割功能开发中...")
    else:
        st.warning("未检测到可分割的物体")


def run_scene_graph(image, detections):
    """场景图生成"""
    sg_gen = get_sg_generator()
    
    # 生成场景图
    graph = sg_gen.generate(image, detections)
    
    # 可视化
    annotated = sg_gen.visualize(image, detections, graph)
    
    st.image(annotated, channels="BGR", use_container_width=True, caption="场景图")
    
    # 显示图结构
    st.subheader("🕸️ 场景图结构")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 节点 (物体)**")
        for node in graph.get("nodes", []):
            st.write(f"- {node['label']} (置信度: {node.get('confidence', 0):.2f})")
    
    with col2:
        st.markdown("**🔗 边 (关系)**")
        for edge in graph.get("edges", []):
            subj = graph["nodes"][edge["subject"]]["label"]
            obj = graph["nodes"][edge["object"]]["label"]
            st.write(f"- {subj} → {edge['relation']} → {obj}")
    
    # 导出JSON
    with st.expander("📥 导出场景图"):
        graph_json = download_json(graph)
        st.download_button(
            "下载JSON",
            graph_json,
            "scene_graph.json",
            "application/json",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
