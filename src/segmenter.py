"""
SAM分割模块 - 增强版
基于Segment Anything Model实现交互式分割
支持多种模型后端
"""
import os
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
from PIL import Image
import streamlit as st


class SAMSegmenter:
    """
    SAM分割器 - 支持多种接入方式
    
    支持模式:
    1. 本地SAM: 使用Meta的SAM模型
    2. Grounded-SAM: 检测+分割一体化
    3. 轻量模式: 使用OpenCV GrabCut等
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化SAM分割器
        
        Args:
            config: 配置字典 {
                "mode": "sam" | "grounded" | "lightweight",
                "model_type": "vit_b" | "vit_l" | "vit_h",
                "model_path": "path/to/model",
                "device": "cuda" | "cpu",
                ...
            }
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "lightweight")
        self.model_type = self.config.get("model_type", "vit_b")
        self.device = self.config.get("device", "cpu")
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载分割模型"""
        if self.mode == "sam":
            self._load_sam_model()
        elif self.mode == "grounded":
            self._load_grounded_sam()
        # lightweight模式使用OpenCV
    
    def _load_sam_model(self):
        """加载Meta SAM模型"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            st.info("正在加载SAM模型...")
            
            # 模型路径
            model_path = self.config.get("model_path")
            if not model_path:
                # 默认路径
                model_path = os.path.join(
                    os.path.dirname(__file__), 
                    "../models/sam_vit_b.pth"
                )
            
            if not os.path.exists(model_path):
                st.warning(f"SAM模型不存在: {model_path}")
                st.info("请从 https://github.com/facebookresearch/segment-anything 下载模型")
                return
            
            # 加载模型
            sam = sam_model_registry[self.model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.model = sam
            
            st.success("SAM模型加载成功!")
            
        except ImportError:
            st.warning("请安装 segment-anything: pip install segment-anything")
            st.info("或使用轻量级模式")
        except Exception as e:
            st.error(f"SAM模型加载失败: {e}")
    
    def _load_grounded_sam(self):
        """加载Grounded-SAM"""
        st.info("Grounded-SAM模式: 结合检测和分割")
        # TODO: 实现Grounded-SAM加载
        # 需要: Grounding DINO + SAM
    
    def _load_lightweight(self):
        """加载轻量级分割"""
        # 使用OpenCV GrabCut
        pass
    
    def segment(self, image: Union[np.ndarray, Image.Image],
                points: Optional[List[Tuple[int, int]]] = None,
                boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                labels: Optional[List[int]] = None) -> List[Dict]:
        """
        分割图像中的物体
        
        Args:
            image: 图像 (np.array 或 PIL.Image)
            points: 提示点 [[x, y], ...]
            boxes: 边界框 [[x1,y1,x2,y2], ...]
            labels: 提示点标签 [1(前景), 0(背景), ...]
            
        Returns:
            list: 分割结果列表 [{
                "mask": np.array,
                "bbox": [x1,y1,x2,y2],
                "area": int
            }, ...]
        """
        # 转换图像格式
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.mode == "sam" and self.predictor:
            return self._sam_segment(image, points, boxes, labels)
        elif self.mode == "grounded":
            return self._grounded_segment(image, boxes)
        else:
            return self._lightweight_segment(image, boxes)
    
    def _sam_segment(self, image: np.ndarray,
                     points: Optional[List[Tuple[int, int]]] = None,
                     boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                     labels: Optional[List[int]] = None) -> List[Dict]:
        """SAM分割"""
        # 设置图像
        self.predictor.set_image(image)
        
        results = []
        
        if points is not None and len(points) > 0:
            # 点提示
            points_array = np.array(points)
            if labels is None:
                labels = np.ones(len(points), dtype=int)
            labels_array = np.array(labels)
            
            masks, scores, _ = self.predictor.predict(
                points=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
            
            for i, mask in enumerate(masks):
                results.append({
                    "mask": mask,
                    "score": scores[i],
                    "area": mask.sum()
                })
        
        elif boxes is not None and len(boxes) > 0:
            # 框提示
            for box in boxes:
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=True
                )
                
                # 取最好的mask
                best_idx = np.argmax(scores)
                results.append({
                    "mask": masks[best_idx],
                    "score": scores[best_idx],
                    "bbox": box,
                    "area": masks[best_idx].sum()
                })
        
        return results
    
    def _grounded_segment(self, image: np.ndarray,
                          boxes: Optional[List[Tuple]] = None) -> List[Dict]:
        """Grounded-SAM分割"""
        # TODO: 实现
        return []
    
    def _lightweight_segment(self, image: np.ndarray,
                             boxes: Optional[List[Tuple]] = None) -> List[Dict]:
        """轻量级分割 - 使用GrabCut"""
        import cv2
        
        results = []
        
        # 如果没有boxes，尝试自动检测
        if boxes is None or len(boxes) == 1:
            # 使用简单的前景检测
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:  # 过滤小区域
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    results.append({
                        "mask": mask > 0,
                        "bbox": [x, y, x+w, y+h],
                        "area": cv2.contourArea(cnt)
                    })
        else:
            # 使用提供的boxes进行GrabCut
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                rect = (x1, y1, x2-x1, y2-y1)
                
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 
                           5, cv2.GC_INIT_WITH_RECT)
                
                # 提取前景
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                results.append({
                    "mask": mask2 > 0,
                    "bbox": box,
                    "area": mask2.sum()
                })
        
        return results
    
    def segment_everything(self, image: Union[np.ndarray, Image.Image],
                          stability_score_thresh: float = 0.95) -> List[Dict]:
        """
        全自动分割 - 生成所有可能的物体
        
        Args:
            image: 图像
            stability_score_thresh: 稳定性阈值
            
        Returns:
            list: 分割结果列表
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.mode == "sam" and self.predictor:
            from segment_anything import SamAutomaticMaskGenerator
            
            st.info("正在生成所有分割...")
            
            # 创建自动分割生成器
            mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                stability_score_thresh=stability_score_thresh
            )
            
            masks = mask_generator.generate(image)
            
            results = []
            for m in masks:
                results.append({
                    "segmentation": m["segmentation"],
                    "bbox": m["bbox"],
                    "area": m["area"],
                    "stability_score": m.get("stability_score", 0)
                })
            
            return results
        else:
            # 轻量模式: 使用轮廓检测
            return self._lightweight_segment(image)
    
    def refine_segment(self, image: np.ndarray, 
                       mask: np.ndarray,
                       points: List[Tuple[int, int]],
                       positive: bool = True) -> np.ndarray:
        """
        优化分割结果
        
        Args:
            image: 图像
            mask: 初始掩码
            points: 优化提示点
            positive: True=添加前景, False=添加背景
            
        Returns:
            np.array: 优化后的掩码
        """
        if self.mode == "sam" and self.predictor:
            self.predictor.set_image(image)
            
            labels = np.ones(len(points), dtype=int) if positive else np.zeros(len(points), dtype=int)
            
            masks, scores, _ = self.predictor.predict(
                points=np.array(points),
                point_labels=labels,
                mask_input=mask[np.newaxis, ...],
                multimask_output=False
            )
            
            return masks[0]
        
        return mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray,
                   color: Tuple[int, int, int] = (255, 0, 0),
                   alpha: float = 0.5) -> np.ndarray:
        """
        将掩码应用到图像上
        
        Args:
            image: 原图
            mask: 掩码
            color: 颜色
            alpha: 透明度
            
        Returns:
            np.array: 带掩码的图像
        """
        import cv2
        
        result = image.copy()
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # 混合
        result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)
        
        # 绘制边界
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)
        
        return result


def create_sam_segmenter(config: Optional[Dict] = None) -> SAMSegmenter:
    """工厂函数: 创建SAM分割器"""
    return SAMSegmenter(config)


# SAM模型配置
SAM_MODELS = {
    "vit_b": {
        "name": "SAM ViT-B",
        "size": "375MB",
        "speed": "快",
        "accuracy": "一般",
        "checkpoint": "sam_vit_b.pth"
    },
    "vit_l": {
        "name": "SAM ViT-L", 
        "size": "1.2GB",
        "speed": "中等",
        "accuracy": "较好",
        "checkpoint": "sam_vit_l.pth"
    },
    "vit_h": {
        "name": "SAM ViT-H",
        "size": "2.4GB",
        "speed": "慢",
        "accuracy": "最好",
        "checkpoint": "sam_vit_h.pth"
    }
}
