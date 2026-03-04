"""
VLM智能理解模块 - 增强版
支持多种VLM接入方式: API / 本地模型 / 轻量级方案
"""
import os
import json
import base64
import requests
from typing import Optional, Dict, List, Union
from PIL import Image
import numpy as np
import streamlit as st


class VLMProcessor:
    """
    VLM处理器 - 支持多种接入方式
    
    支持模式:
    1. API模式: OpenAI GPT-4V, Anthropic Claude, 等
    2. 本地模型: LLaVA, Qwen-VL, 等
    3. 轻量模式: CLIP + 规则
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化VLM处理器
        
        Args:
            config: 配置字典 {
                "mode": "api" | "local" | "lightweight",
                "provider": "openai" | "anthropic" | "qwen",
                "api_key": "xxx",
                "model_path": "path/to/model",
                ...
            }
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "lightweight")
        self.provider = self.config.get("provider", "openai")
        self.client = None
        self.model = None
        self._init_client()
    
    def _init_client(self):
        """初始化VLM客户端"""
        if self.mode == "api":
            self._init_api_client()
        elif self.mode == "local":
            self._init_local_model()
        # lightweight模式不需要初始化
    
    def _init_api_client(self):
        """初始化API客户端"""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        
        if self.provider == "openai" and api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.model = self.config.get("model", "gpt-4o")
            except ImportError:
                st.warning("请安装 openai: pip install openai")
        
        elif self.provider == "anthropic":
            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                # Anthropic需要单独处理
                self.client = {"api_key": api_key, "provider": "anthropic"}
                self.model = self.config.get("model", "claude-3-opus-20240229")
        
        elif self.provider == "qwen":
            api_key = self.config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
            if api_key:
                self.client = {"api_key": api_key, "provider": "qwen"}
                self.model = self.config.get("model", "qwen-vl-max")
    
    def _init_local_model(self):
        """初始化本地模型"""
        # TODO: 实现本地模型加载
        # 可以使用: LLaVA, Qwen-VL, MiniGPT-4等
        model_path = self.config.get("model_path", "")
        st.info(f"本地模型加载: {model_path}")
    
    def describe(self, image: Union[Image.Image, np.ndarray], 
                 prompt: Optional[str] = None) -> str:
        """
        生成图像描述
        
        Args:
            image: PIL.Image 或 np.array
            prompt: 可选的提示词
            
        Returns:
            str: 图像描述
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        default_prompt = prompt or "请详细描述这张图片的内容，包括场景、物体、颜色、动作等。"
        
        if self.mode == "api" and self.client:
            return self._api_describe(image, default_prompt)
        elif self.mode == "local" and self.model:
            return self._local_describe(image, default_prompt)
        else:
            return self._lightweight_describe(image, default_prompt)
    
    def _api_describe(self, image: Image.Image, prompt: str) -> str:
        """API方式描述"""
        if hasattr(self, 'client') and hasattr(self.client, 'chat'):
            # OpenAI
            return self._openai_describe(image, prompt)
        elif isinstance(self.client, dict):
            provider = self.client.get("provider")
            if provider == "anthropic":
                return self._anthropic_describe(image, prompt)
            elif provider == "qwen":
                return self._qwen_describe(image, prompt)
        
        return "API客户端未正确初始化"
    
    def _openai_describe(self, image: Image.Image, prompt: str) -> str:
        """OpenAI GPT-4V 描述"""
        try:
            # 将图片转为base64
            import io
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API调用失败: {str(e)}"
    
    def _anthropic_describe(self, image: Image.Image, prompt: str) -> str:
        """Anthropic Claude VLM 描述"""
        # Claude不支持直接图像输入，需要转换
        return "Anthropic Claude当前不支持图像输入，请使用OpenAI GPT-4V"
    
    def _qwen_describe(self, image: Image.Image, prompt: str) -> str:
        """阿里Qwen-VL 描述"""
        try:
            import io
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            import dashscope
            from dashscope import MultiModalConversation
            
            api_key = self.client.get("api_key")
            dashscope.api_key = api_key
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{img_b64}"},
                        {"text": prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            return response.output.choices[0].message.content[0]["text"]
        except Exception as e:
            return f"Qwen-VL API调用失败: {str(e)}"
    
    def _local_describe(self, image: Image.Image, prompt: str) -> str:
        """本地模型描述"""
        # TODO: 实现本地模型调用
        return "本地模型功能开发中..."
    
    def _lightweight_describe(self, image: Image.Image, prompt: str) -> str:
        """
        轻量级描述 - 基于规则和统计
        不需要额外模型
        """
        # 获取图片基本信息
        width, height = image.size
        
        # 简单分析
        analysis = []
        analysis.append(f"图像尺寸: {width}x{height}")
        
        # 颜色分析
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            mean_colors = img_array.mean(axis=(0,1))
            r, g, b = mean_colors
            if r > g and r > b:
                analysis.append("主色调: 红色系")
            elif g > r and g > b:
                analysis.append("主色调: 绿色系")
            elif b > r and b > g:
                analysis.append("主色调: 蓝色系")
            else:
                analysis.append("主色调: 中性色系")
        
        # 返回分析结果
        result = "📷 图像分析:\n" + "\n".join(f"- {a}" for a in analysis)
        result += "\n\n💡 提示: 配置API密钥可启用GPT-4V智能理解"
        
        return result
    
    def answer(self, image: Union[Image.Image, np.ndarray], 
               question: str) -> str:
        """
        回答关于图像的问题
        
        Args:
            image: PIL.Image 或 np.array
            question: str, 问题
            
        Returns:
            str: 回答
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        prompt = f"请根据图片回答以下问题: {question}"
        
        if self.mode == "api" and self.client:
            return self._api_describe(image, prompt)
        elif self.mode == "local" and self.model:
            return self._local_describe(image, prompt)
        else:
            return "⚠️ 请配置API密钥启用智能问答功能\n\n支持的提供商: OpenAI GPT-4V, 阿里Qwen-VL"
    
    def detect_and_describe(self, image: Union[Image.Image, np.ndarray], 
                           detections: List[Dict]) -> Dict[str, str]:
        """
        对检测到的物体进行描述
        
        Args:
            image: PIL.Image 或 np.array
            detections: 检测结果列表
            
        Returns:
            dict: {物体类别: 描述}
        """
        descriptions = {}
        
        # 按类别分组
        classes = set(d.get("class", "") for d in detections)
        
        for cls in classes:
            prompt = f"描述图片中的这个{cls}物体，它的特征、颜色、状态等"
            descriptions[cls] = self.describe(image, prompt)
        
        return descriptions
    
    def batch_describe(self, images: List[Image.Image]) -> List[str]:
        """
        批量描述多张图片
        
        Args:
            images: 图片列表
            
        Returns:
            list: 描述列表
        """
        return [self.describe(img) for img in images]


def create_vlm_processor(config: Optional[Dict] = None) -> VLMProcessor:
    """
    工厂函数: 创建VLM处理器
    
    Args:
        config: 配置字典
        
    Returns:
        VLMProcessor实例
    """
    # 从环境变量读取配置
    if config is None:
        config = {}
    
    # 自动检测可用模式
    if os.getenv("OPENAI_API_KEY"):
        config.setdefault("mode", "api")
        config.setdefault("provider", "openai")
    elif os.getenv("DASHSCOPE_API_KEY"):
        config.setdefault("mode", "api")
        config.setdefault("provider", "qwen")
    
    return VLMProcessor(config)


# 默认配置示例
DEFAULT_CONFIG = {
    # 模式: api / local / lightweight
    "mode": "lightweight",
    
    # API提供商 (mode=api时有效)
    "provider": "openai",  # openai / anthropic / qwen
    
    # API密钥 (可从环境变量读取)
    # "api_key": os.getenv("OPENAI_API_KEY"),
    
    # 模型名称
    "model": "gpt-4o",  # openai: gpt-4o, gpt-4-vision-preview
                       # qwen: qwen-vl-max, qwen-vl-plus
    
    # 本地模型路径 (mode=local时有效)
    # "model_path": "./models/llava-1.5-7b"
}
