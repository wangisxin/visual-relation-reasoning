#!/bin/bash
# 本地测试运行脚本
# 在项目目录下执行: bash test_run.sh

echo "========================================="
echo "多维图像识别与理解系统 - 本地测试"
echo "========================================="

# 检查Python环境
echo "[1/5] 检查Python环境..."
python3 --version || { echo "❌ Python未安装"; exit 1; }

# 创建虚拟环境 (可选)
if [ ! -d "venv" ]; then
    echo "[2/5] 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "[3/5] 安装依赖..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install streamlit -q

# 检查模型文件
echo "[4/5] 检查模型文件..."
if [ ! -f "src/yolov8n.pt" ]; then
    echo "⚠️ 模型文件不存在，将自动下载..."
fi

# 运行测试
echo "[5/5] 运行测试..."
echo ""
echo "测试1: 模块导入测试"
python3 -c "
from detector import ObjectDetector
from relation_reasoner import RelationReasoner
from vlm import VLMProcessor
from segmenter import SAMSegmenter
from scene_graph import SceneGraphGenerator
print('✅ 所有模块导入成功')
"

echo ""
echo "测试2: 检测器初始化"
python3 -c "
from detector import ObjectDetector
detector = ObjectDetector('n')
print('✅ 检测器初始化成功')
"

echo ""
echo "测试3: Streamlit UI"
echo "运行: streamlit run src/app.py"
echo ""

# 提示用户
echo "========================================="
echo "✅ 测试完成!"
echo "如需运行Web界面，请执行:"
echo "  streamlit run src/app.py"
echo "========================================="
