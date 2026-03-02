"""
物体检测模块 - 基于 YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple


class ObjectDetector:
    """YOLOv8 物体检测器"""
    
    # COCO 80 类名称
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_size: str = 'n'):
        """
        初始化检测器
        
        Args:
            model_size: 'n'(nano), 's'(small), 'm'(medium), 'l'(large)
        """
        self.model_size = model_size
        # 预生成颜色盘，避免重复计算
        self._color_cache = {}
        print(f"加载 YOLOv8{model_size} 模型...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("模型加载完成")
    
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表，每项包含: class_id, class_name, confidence, bbox
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': self.CLASSES[cls_id],
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
        
        return detections
    
    def draw_detections(self, image_path: str, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image_path: 图像路径
            detections: 检测结果
            
        Returns:
            绘制后的图像
        """
        img = cv2.imread(image_path)
        
        colors = self._generate_colors(len(detections))
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = colors[i]
            
            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """生成 n 个随机颜色（带缓存）"""
        if n in self._color_cache:
            return self._color_cache[n]
        
        np.random.seed(42)
        colors = [(np.random.randint(50, 255), 
                   np.random.randint(50, 255), 
                   np.random.randint(50, 255)) for _ in range(n)]
        self._color_cache[n] = colors
        return colors


if __name__ == '__main__':
    # 测试
    detector = ObjectDetector('n')
    
    # 下载测试图片
    import urllib.request
    test_url = 'https://ultralytics.com/images/bus.jpg'
    test_path = 'test.jpg'
    urllib.request.urlretrieve(test_url, test_path)
    
    # 检测
    detections = detector.detect(test_path)
    print(f"检测到 {len(detections)} 个物体:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
