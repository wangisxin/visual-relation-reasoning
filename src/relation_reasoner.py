"""
关系推理模块 - 空间关系、语义关系、计数
"""
from typing import List, Dict, Tuple
import numpy as np


class RelationReasoner:
    """关系推理器"""
    
    def __init__(self):
        # 语义关系规则库
        self.semantic_rules = self._build_semantic_rules()
    
    def reason(self, detections: List[Dict]) -> Dict:
        """
        推理图像中的关系
        
        Args:
            detections: 物体检测结果
            
        Returns:
            包含 spatial_relations, semantic_relations, counts 的字典
        """
        # 空间关系
        spatial_relations = self._compute_spatial_relations(detections)
        
        # 语义关系
        semantic_relations = self._compute_semantic_relations(detections, spatial_relations)
        
        # 计数
        counts = self._count_objects(detections)
        
        return {
            'spatial_relations': spatial_relations,
            'semantic_relations': semantic_relations,
            'counts': counts
        }
    
    def _compute_spatial_relations(self, detections: List[Dict]) -> List[Dict]:
        """计算空间关系"""
        relations = []
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                obj1 = detections[i]
                obj2 = detections[j]
                
                relation = self._get_spatial_relation(obj1, obj2)
                if relation:
                    relations.append({
                        'object1': obj1['class_name'],
                        'object2': obj2['class_name'],
                        'relation': relation,
                        'type': 'spatial'
                    })
        
        return relations
    
    def _get_spatial_relation(self, obj1: Dict, obj2: Dict) -> str:
        """判断两个物体之间的空间关系"""
        x1_1, y1_1, x2_1, y2_1 = obj1['bbox']
        x1_2, y1_2, x2_2, y2_2 = obj2['bbox']
        
        # 计算中心点
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        
        # 计算面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 计算 IoU
        iou = self._compute_iou(obj1['bbox'], obj2['bbox'])
        
        # 判断遮挡关系 (IoU > 0.5)
        if iou > 0.5:
            # 一个框是否包含另一个
            if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
                return f"{obj1['class_name']} 遮挡了 {obj2['class_name']}"
            elif x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x1_1 and y2_2 >= y1_1:
                return f"{obj2['class_name']} 遮挡了 {obj1['class_name']}"
        
        # 上下关系
        if cy1 < cy2 - 10:
            return f"{obj1['class_name']} 在 {obj2['class_name']} 上方"
        elif cy2 < cy1 - 10:
            return f"{obj2['class_name']} 在 {obj1['class_name']} 上方"
        
        # 左右关系
        if cx1 < cx2 - 10:
            return f"{obj1['class_name']} 在 {obj2['class_name']} 左侧"
        elif cx2 < cx1 - 10:
            return f"{obj2['class_name']} 在 {obj1['class_name']} 左侧"
        
        # 前后关系 (基于面积)
        if area1 < area1 * 0.8 and area2 > area1 * 1.2:
            return f"{obj1['class_name']} 在 {obj2['class_name']} 前面"
        elif area2 < area1 * 0.8 and area1 > area2 * 1.2:
            return f"{obj2['class_name']} 在 {obj1['class_name']} 前面"
        
        # 相邻/重叠
        if iou > 0.1:
            return f"{obj1['class_name']} 与 {obj2['class_name']} 相邻"
        
        return None
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集区域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_semantic_relations(self, detections: List[Dict], 
                                    spatial_relations: List[Dict]) -> List[Dict]:
        """计算语义关系"""
        relations = []
        
        # 获取所有物体类别
        classes = [d['class_name'] for d in detections]
        
        # 应用语义规则
        for rule in self.semantic_rules:
            subjects = rule['subjects']
            objects = rule['objects']
            
            # 检查是否有匹配的类别
            for i, det1 in enumerate(detections):
                if det1['class_name'] not in subjects:
                    continue
                    
                for j, det2 in enumerate(detections):
                    if i == j:
                        continue
                    if det2['class_name'] not in objects:
                        continue
                    
                    # 检查空间位置是否支持这个语义关系
                    relation_text = rule['template'].format(
                        subject=det1['class_name'],
                        obj=det2['class_name']
                    )
                    
                    relations.append({
                        'object1': det1['class_name'],
                        'object2': det2['class_name'],
                        'relation': relation_text,
                        'type': 'semantic'
                    })
        
        # 去重
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['object1'], rel['object2'], rel['relation'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations[:10]  # 限制数量
    
    def _build_semantic_rules(self) -> List[Dict]:
        """构建语义关系规则库"""
        return [
            # 人与物品的关系
            {
                'subjects': ['person'],
                'objects': ['backpack', 'handbag', 'tie', 'suitcase', 'bottle', 
                           'cup', 'fork', 'knife', 'spoon', 'bowl', 'cell phone',
                           'book', 'clock', 'vase', 'teddy bear'],
                'template': '人拿着 {obj}'
            },
            {
                'subjects': ['person'],
                'objects': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
                'template': '人坐在 {obj} 上'
            },
            {
                'subjects': ['person'],
                'objects': ['sports ball', 'kite', 'frisbee', 'skateboard', 
                           'surfboard', 'tennis racket', 'baseball bat', 'baseball glove'],
                'template': '人在玩 {obj}'
            },
            {
                'subjects': ['person'],
                'objects': ['tv', 'laptop', 'keyboard', 'mouse', 'microwave', 'oven'],
                'template': '人在使用 {obj}'
            },
            # 动物与物品
            {
                'subjects': ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 
                           'elephant', 'bear', 'zebra', 'giraffe'],
                'objects': ['bench', 'bed', 'couch'],
                'template': '{subject} 躺在 {obj} 上'
            },
            # 交通工具
            {
                'subjects': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                'objects': ['person'],
                'template': '{subject} 附近有 {obj}'
            },
            # 食物关系
            {
                'subjects': ['person'],
                'objects': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                           'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
                'template': '人在吃 {obj}'
            },
            {
                'subjects': ['cat', 'dog'],
                'objects': ['bottle', 'bowl'],
                'template': '{subject} 在喝 {obj} 里的东西'
            },
        ]
    
    def _count_objects(self, detections: List[Dict]) -> Dict[str, int]:
        """统计各类别数量"""
        counts = {}
        for det in detections:
            name = det['class_name']
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    # 测试
    reasoner = RelationReasoner()
    
    test_detections = [
        {'class_name': 'person', 'bbox': (100, 100, 200, 300), 'confidence': 0.9},
        {'class_name': 'chair', 'bbox': (80, 250, 220, 400), 'confidence': 0.8},
        {'class_name': 'bottle', 'bbox': (120, 200, 140, 240), 'confidence': 0.7},
    ]
    
    result = reasoner.reason(test_detections)
    print("空间关系:", result['spatial_relations'])
    print("语义关系:", result['semantic_relations'])
    print("计数:", result['counts'])
