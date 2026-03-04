"""
场景图生成模块 - 增强版
基于检测结果生成物体-关系-物体的图结构
支持多种关系推理方法
"""
import os
import json
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import streamlit as st


class SceneGraphGenerator:
    """
    场景图生成器
    
    支持模式:
    1. rule: 基于规则的关系推理
    2. llm: 基于大语言模型的关系推理
    3. neural: 基于神经网络的关系推理
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化场景图生成器
        
        Args:
            config: 配置字典 {
                "mode": "rule" | "llm" | "neural",
                "llm_client": VLM客户端,
                "relation_types": ["spatial", "semantic"],
                ...
            }
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "rule")
        self.llm_client = self.config.get("llm_client")
        self.relation_types = self.config.get("relation_types", 
                                               ["spatial", "semantic"])
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        if self.mode == "neural":
            # 加载神经网络模型
            st.info("场景图神经网络模式初始化中...")
    
    def generate(self, image: Union[np.ndarray, Image.Image],
                 detections: List[Dict],
                 include_visual: bool = True) -> Dict:
        """
        生成场景图
        
        Args:
            image: 图像
            detections: 检测结果列表
            include_visual: 是否包含视觉属性
            
        Returns:
            dict: 场景图 {
                "nodes": [{"id", "label", "bbox", "attributes"}, ...],
                "edges": [{"subject", "relation", "object", "type"}, ...],
                "metadata": {}
            }
        """
        # 转换图像格式
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if not detections:
            return {"nodes": [], "edges": [], "metadata": {}}
        
        nodes = []
        edges = []
        
        # 创建节点
        for i, det in enumerate(detections):
            node = {
                "id": i,
                "label": det.get("class", "object"),
                "bbox": det.get("bbox", []),
                "confidence": det.get("confidence", 0.0)
            }
            
            # 添加视觉属性
            if include_visual:
                node["attributes"] = self._extract_attributes(image, det)
            
            nodes.append(node)
        
        # 提取关系
        if self.mode == "llm" and self.llm_client:
            edges = self._llm_extract_relations(image, detections)
        elif self.mode == "neural":
            edges = self._neural_extract_relations(image, detections)
        else:
            edges = self._rule_extract_relations(detections)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "num_objects": len(nodes),
                "num_relations": len(edges),
                "image_size": image.shape[:2]
            }
        }
    
    def _extract_attributes(self, image: np.ndarray, detection: Dict) -> Dict:
        """
        提取物体视觉属性
        
        Args:
            image: 图像
            detection: 检测结果
            
        Returns:
            dict: 属性字典
        """
        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            return {}
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 裁剪物体区域
        obj_roi = image[y1:y2, x1:x2]
        if obj_roi.size == 0:
            return {}
        
        # 颜色分析
        mean_color = obj_roi.mean(axis=(0,1))
        std_color = obj_roi.std(axis=(0,1))
        
        # 主色调判断
        r, g, b = mean_color
        if r > g + 30 and r > b + 30:
            color = "red"
        elif g > r + 30 and g > b + 30:
            color        elif b > r + 30 = "green"
 and b > g + 30:
            color = "blue"
        elif r > 200 and g > 200 and b > 200:
            color = "white"
        elif r < 50 and g < 50 and b < 50:
            color = "black"
        else:
            color = "other"
        
        # 大小
        area = (x2 - x1) * (y2 - y1)
        if area > 50000:
            size = "large"
        elif area > 10000:
            size = "medium"
        else:
            size = "small"
        
        return {
            "color": color,
            "size": size,
            "dominant_color": {
                "r": float(r), "g": float(g), "b": float(b)
            }
        }
    
    def _rule_extract_relations(self, detections: List[Dict]) -> List[Dict]:
        """
        基于规则的关系提取
        
        Args:
            detections: 检测结果
            
        Returns:
            list: 关系列表
        """
        relations = []
        n = len(detections)
        
        for i in range(n):
            for j in range(i+1, n):
                # 空间关系
                spatial_rel = self._compute_spatial_relation(
                    detections[i], detections[j]
                )
                
                if spatial_rel:
                    relations.append({
                        "subject": i,
                        "object": j,
                        "relation": spatial_rel,
                        "type": "spatial",
                        "confidence": 0.9
                    })
                
                # 语义关系
                semantic_rel = self._compute_semantic_relation(
                    detections[i], detections[j]
                )
                
                if semantic_rel:
                    relations.append({
                        "subject": i,
                        "object": j,
                        "relation": semantic_rel,
                        "type": "semantic",
                        "confidence": 0.7
                    })
        
        return relations
    
    def _compute_spatial_relation(self, obj1: Dict, obj2: Dict) -> Optional[str]:
        """
        计算空间关系
        
        Args:
            obj1, obj2: 检测结果
            
        Returns:
            str: 空间关系
        """
        bbox1 = obj1.get("bbox", [])
        bbox2 = obj2.get("bbox", [])
        
        if len(bbox1) != 4 or len(bbox2) != 4:
            return None
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 中心点
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # 判断关系
        # 上下关系
        if cy1 < cy2 - (y2_1 - y1_1) * 0.3:
            return "above"
        elif cy2 < cy1 - (y2_2 - y1_2) * 0.3:
            return "below"
        
        # 左右关系
        if cx1 < cx2 - (x2_1 - x1_1) * 0.3:
            return "left"
        elif cx2 < cx1 - (x2_2 - x1_2) * 0.3:
            return "right"
        
        # 包含关系
        if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
            return "inside"
        elif x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1:
            return "contains"
        
        # 遮挡关系 (重叠)
        overlap = self._compute_iou(bbox1, bbox2)
        if overlap > 0.3:
            return "overlaps"
        
        return None
    
    def _compute_semantic_relation(self, obj1: Dict, obj2: Dict) -> Optional[str]:
        """
        计算语义关系
        
        Args:
            obj1, obj2: 检测结果
            
        Returns:
            str: 语义关系
        """
        class1 = obj1.get("class", "").lower()
        class2 = obj2.get("class", "").lower()
        
        # 人物相关关系
        person_classes = ["person", "man", "woman", "child", "boy", "girl"]
        
        if class1 in person_classes:
            if class2 in ["phone", "cell phone", "mobile phone"]:
                return "holding"
            elif class2 in ["chair", "sofa", "seat"]:
                return "sitting on"
            elif class2 in ["table", "desk"]:
                return "using"
            elif class2 in ["book", "newspaper", "laptop", "computer"]:
                return "reading"
        
        if class2 in person_classes:
            if class1 in ["phone", "cell phone"]:
                return "held by"
            elif class1 in ["chair", "sofa"]:
                return "sat by"
        
        # 物体相关
        if class1 in ["car", "vehicle", "truck", "bus"] and \
           class2 in ["road", "street"]:
            return "on"
        
        if class1 in ["bottle", "cup", "glass"] and \
           class2 in ["table", "desk"]:
            return "on"
        
        return None
    
    def _compute_iou(self, bbox1: List, bbox2: List) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
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
    
    def _llm_extract_relations(self, image: np.ndarray, 
                               detections: List[Dict]) -> List[Dict]:
        """基于LLM的关系提取"""
        if not self.llm_client:
            return self._rule_extract_relations(detections)
        
        # 构建提示
        objects = [f"{i}: {d.get('class', 'object')}" 
                   for i, d in enumerate(detections)]
        
        prompt = f"""分析以下物体之间的关系:
物体: {', '.join(objects)}

请输出物体之间的关系，格式:
(物体1) - [关系] -> (物体2)

只输出重要的关系，不要输出太多。"""
        
        # 调用LLM
        result = self.llm_client.answer(image, prompt)
        
        # 解析结果
        # TODO: 实现解析
        return []
    
    def _neural_extract_relations(self, image: np.ndarray,
                                   detections: List[Dict]) -> List[Dict]:
        """基于神经网络的关系提取"""
        # TODO: 实现神经网络方法
        # 可使用: RelTR, MSDN, GraphRCNN等
        return self._rule_extract_relations(detections)
    
    def visualize(self, image: Union[np.ndarray, Image.Image],
                  detections: List[Dict],
                  graph: Dict,
                  show_attributes: bool = True) -> np.ndarray:
        """
        可视化场景图
        
        Args:
            image: 图像
            detections: 检测结果
            graph: 场景图
            show_attributes: 是否显示属性
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        result = image.copy()
        
        # 颜色映射
        spatial_color = (0, 255, 255)  # 青色 - 空间关系
        semantic_color = (255, 0, 255)  # 紫色 - 语义关系
        
        # 绘制边界框
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 标签
                label = det.get("class", "object")
                cv2.putText(result, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制关系边
        for edge in graph.get("edges", []):
            subj = graph["nodes"][edge["subject"]]
            obj = graph["nodes"][edge["object"]]
            
            subj_bbox = subj.get("bbox", [])
            obj_bbox = obj.get("bbox", [])
            
            if len(subj_bbox) == 4 and len(obj_bbox) == 4:
                # 中心点
                sx = int((subj_bbox[0] + subj_bbox[2]) / 2)
                sy = int((subj_bbox[1] + subj_bbox[3]) / 2)
                ox = int((obj_bbox[0] + obj_bbox[2]) / 2)
                oy = int((obj_bbox[1] + obj_bbox[3]) / 2)
                
                # 颜色
                color = spatial_color if edge.get("type") == "spatial" else semantic_color
                
                # 画线
                cv2.arrowedLine(result, (sx, sy), (ox, oy), color, 2, 
                               tipLength=0.2)
                
                # 关系标签
                mx, my = (sx + ox) // 2, (sy + oy) // 2
                cv2.putText(result, edge["relation"], (mx, my-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加图例
        legend_y = 30
        cv2.putText(result, "空间关系", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, spatial_color, 1)
        cv2.putText(result, "语义关系", (10, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, semantic_color, 1)
        
        return result
    
    def export_json(self, graph: Dict, filepath: str):
        """导出JSON格式"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
    
    def export_graphviz(self, graph: Dict, filepath: str):
        """导出Graphviz格式"""
        lines = ["digraph SceneGraph {"]
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=rounded];')
        
        # 节点
        for node in graph["nodes"]:
            label = node.get("label", "object")
            attrs = node.get("attributes", {})
            if attrs:
                color = attrs.get("color", "white")
                label = f"{label}\\n({color})"
            
            lines.append(f'  n{node["id"]} [label="{label}"];')
        
        # 边
        for edge in graph["edges"]:
            style = "solid" if edge.get("type") == "spatial" else "dashed"
            lines.append(f'  n{edge["subject"]} -> n{edge["object"]} '
                        f'[label="{edge["relation"]}", style={style}];')
        
        lines.append("}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    def export_cytoscape(self, graph: Dict, filepath: str):
        """导出Cytoscape/网络图格式"""
        cytoscape_data = {
            "data": graph.get("metadata", {}),
            "elements": {
                "nodes": [
                    {"data": {"id": f"n{n['id']}", "label": n['label']}}
 graph["nodes"]
                    for n in                ],
                "edges": [
                    {"data": {
                        "source": f"n{e['subject']}", 
                        "target": f"n{e['object']}",
                        "label": e['relation']
                    }}
                    for e in graph["edges"]
                ]
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cytoscape_data, f, indent=2, ensure_ascii=False)


# 关系类型定义
RELATION_TYPES = {
    "spatial": [
        "above", "below", "left", "right",
        "inside", "outside", "near", "far",
        "front", "behind", "overlaps", "contains"
    ],
    "semantic": [
        "holding", "wearing", "sitting_on", "standing_on",
        "using", "eating", "playing", "looking_at",
        "attached_to", "belong_to", "part_of",
        "on", "near"
    ]
}
