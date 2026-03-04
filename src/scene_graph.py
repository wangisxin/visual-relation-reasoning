"""
场景图生成模块
基于检测结果生成物体-关系-物体的图结构
"""
import streamlit as st
import numpy as np
import json


class SceneGraphGenerator:
    """场景图生成器"""
    
    def __init__(self):
        """初始化"""
        self.relation_extractor = None
        self.visualizer = None
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        st.info("场景图生成模块初始化中...")
        # TODO: 加载关系抽取模型
        # 可选: LLM4SGG, RelTR等
    
    def generate(self, image, detections):
        """
        生成场景图
        
        Args:
            image: np.array, 图像
            detections: list, 检测结果
            
        Returns:
            dict: 场景图 {
                "nodes": [{"id", "label", "bbox"}, ...],
                "edges": [{"subject", "relation", "object"}, ...]
            }
        """
        if not detections:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        # 创建节点
        for i, det in enumerate(detections):
            nodes.append({
                "id": i,
                "label": det.get("class", "object"),
                "bbox": det.get("bbox", []),
                "confidence": det.get("confidence", 0.0)
            })
        
        # 提取关系
        edges = self._extract_relations(detections)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _extract_relations(self, detections):
        """
        提取物体间关系
        
        Args:
            detections: list, 检测结果
            
        Returns:
            list: 关系列表
        """
        relations = []
        n = len(detections)
        
        # 简单的两两关系分析
        # TODO: 集成更复杂的模型
        for i in range(n):
            for j in range(i+1, n):
                rel = self._predict_relation(detections[i], detections[j])
                if rel:
                    relations.append({
                        "subject": i,
                        "object": j,
                        "relation": rel
                    })
        
        return relations
    
    def _predict_relation(self, obj1, obj2):
        """
        预测两个物体的关系
        
        Args:
            obj1, obj2: 检测结果
            
        Returns:
            str: 关系类型
        """
        # TODO: 实现更智能的关系预测
        # 可以使用:
        # 1. 规则方法 (基于位置)
        # 2. 神经网络方法 (RelTR, MSDN等)
        # 3. LLM方法 (LLM4SGG)
        return None
    
    def visualize(self, image, detections, graph):
        """
        可视化场景图
        
        Args:
            image: np.array
            detections: list
            graph: dict, 场景图
            
        Returns:
            np.array: 可视化后的图像
        """
        import cv2
        
        result = image.copy()
        
        # 绘制边界框
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, det.get("class", ""), (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制关系线
        for edge in graph.get("edges", []):
            subj = graph["nodes"][edge["subject"]]
            obj = graph["nodes"][edge["object"]]
            
            if subj["bbox"] and obj["bbox"]:
                # 计算中心点
                sx = int((subj["bbox"][0] + subj["bbox"][2]) / 2)
                sy = int((subj["bbox"][1] + subj["bbox"][3]) / 2)
                ox = int((obj["bbox"][0] + obj["bbox"][2]) / 2)
                oy = int((obj["bbox"][1] + obj["bbox"][3]) / 2)
                
                # 画线
                cv2.line(result, (sx, sy), (ox, oy), (255, 0, 0), 2)
                
                # 画关系标签
                mx, my = (sx + ox) // 2, (sy + oy) // 2
                cv2.putText(result, edge["relation"], (mx, my),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return result
    
    def export_json(self, graph, filepath):
        """
        导出场景图为JSON
        
        Args:
            graph: dict
            filepath: str
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
    
    def export_graphviz(self, graph, filepath):
        """
        导出为Graphviz格式
        
        Args:
            graph: dict
            filepath: str
        """
        lines = ["digraph G {"]
        
        # 节点
        for node in graph["nodes"]:
            label = f'{node["id"]}: {node["label"]}'
            lines.append(f'  {node["id"]} [label="{label}"];')
        
        # 边
        for edge in graph["edges"]:
            lines.append(f'  {edge["subject"]} -> {edge["object"]} [label="{edge[\"relation\"]}"];')
        
        lines.append("}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


# 关系类型定义
RELATION_TYPES = {
    "spatial": [
        "above", "below", "left", "right",
        "inside", "outside", "near", "far",
        "front", "behind", "overlap"
    ],
    "semantic": [
        "holding", "wearing", "sitting_on", "standing_on",
        "using", "eating", "playing", "looking_at",
        "attached_to", "belong_to", "part_of"
    ],
    "action": [
        "running", "walking", "jumping", "sleeping",
        "talking", "reading", "writing", "driving"
    ]
}
