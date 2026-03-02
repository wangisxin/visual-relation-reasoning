"""
图像关系推理系统 - 命令行测试版
"""
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import ObjectDetector
from relation_reasoner import RelationReasoner


def main():
    if len(sys.argv) < 2:
        print("用法: python cli.py <图片路径>")
        print("示例: python cli.py test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"图像关系推理系统 - CLI 测试版")
    print(f"{'='*50}")
    print(f"\n输入图片: {image_path}")
    
    # 初始化
    print("\n[1] 加载模型...")
    detector = ObjectDetector('n')
    reasoner = RelationReasoner()
    print("    模型加载完成")
    
    # 物体检测
    print("\n[2] 物体检测...")
    detections = detector.detect(image_path)
    print(f"    检测到 {len(detections)} 个物体:")
    for det in detections:
        print(f"      - {det['class_name']}: {det['confidence']:.2f}")
    
    # 关系推理
    print("\n[3] 关系推理...")
    relations = reasoner.reason(detections)
    
    # 空间关系
    if relations['spatial_relations']:
        print("    空间关系:")
        for rel in relations['spatial_relations']:
            print(f"      • {rel['relation']}")
    else:
        print("    空间关系: 无")
    
    # 语义关系
    if relations['semantic_relations']:
        print("    语义关系:")
        for rel in relations['semantic_relations']:
            print(f"      • {rel['relation']}")
    else:
        print("    语义关系: 无")
    
    # 计数
    if relations['counts']:
        print("    物体计数:")
        for name, count in relations['counts'].items():
            print(f"      - {name}: {count}个")
    
    # 保存标注图
    print("\n[4] 生成标注图...")
    annotated = detector.draw_detections(image_path, detections)
    output_path = os.path.splitext(image_path)[0] + '_annotated.jpg'
    import cv2
    cv2.imwrite(output_path, annotated)
    print(f"    已保存: {output_path}")
    
    print("\n" + "="*50)
    print("✅ 测试完成!")
    print("="*50)


if __name__ == '__main__':
    import cv2
    main()
