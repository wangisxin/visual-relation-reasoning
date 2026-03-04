"""
性能优化模块
提供模型加速和推理优化功能
"""
import os
import time
import threading
from typing import Optional, Dict, Any
from functools import lru_cache
import streamlit as st


class PerformanceOptimizer:
    """
    性能优化器
    
    功能:
    - 模型缓存
    - 批量推理
    - ONNX加速
    - 异步处理
    """
    
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(__file__), "../models")
        self._model_cache = {}
        self._enable_onnx = False
        self._enable_half_precision = False
        
    def enable_onnx(self):
        """启用ONNX加速"""
        try:
            import onnxruntime
            self._enable_onnx = True
            st.success("ONNX加速已启用")
        except ImportError:
            st.warning("ONNX未安装，加速功能不可用")
    
    def enable_half_precision(self):
        """启用半精度推理 (GPU)"""
        self._enable_half_precision = True
    
    def get_optimal_batch_size(self, image_size: tuple) -> int:
        """
        根据图像尺寸计算最佳批量大小
        
        Args:
            image_size: (width, height)
            
        Returns:
            int: 批量大小
        """
        width, height = image_size
        pixels = width * height
        
        # 根据像素数计算
        if pixels < 640 * 640:
            return 8
        elif pixels < 1280 * 720:
            return 4
        elif pixels < 1920 * 1080:
            return 2
        else:
            return 1
    
    def warmup_model(self, model, sample_input):
        """
        模型预热
        
        Args:
            model: 模型对象
            sample_input: 样本输入
        """
        # 执行一次推理预热
        _ = model(sample_input)
    
    def async_inference(self, model, inputs, callback):
        """
        异步推理
        
        Args:
            model: 模型
            inputs: 输入数据
            callback: 回调函数
        """
        def run():
            result = model(inputs)
            callback(result)
        
        thread = threading.Thread(target=run)
        thread.start()
        return thread


class ModelCache:
    """
    模型缓存管理器
    """
    
    def __init__(self, max_size: int = 3):
        self._cache = {}
        self._max_size = max_size
        self._access_time = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存的模型"""
        if key in self._cache:
            self._access_time[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, model: Any):
        """缓存模型"""
        # 如果缓存已满，删除最老的
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._access_time, key=self._access_time.get)
            del self._cache[oldest_key]
            del self._access_time[oldest_key]
        
        self._cache[key] = model
        self._access_time[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_time.clear()


class InferenceTimer:
    """
    推理计时器
    """
    
    def __init__(self):
        self.timings = {}
    
    def start(self, name: str):
        """开始计时"""
        self.timings[name] = {"start": time.time(), "end": None}
    
    def end(self, name: str):
        """结束计时"""
        if name in self.timings:
            self.timings[name]["end"] = time.time()
    
    def get_time(self, name: str) -> float:
        """获取耗时 (秒)"""
        if name in self.timings:
            start = self.timings[name]["start"]
            end = self.timings[name].get("end", time.time())
            return end - start
        return 0
    
    def get_fps(self, name: str) -> float:
        """获取FPS"""
        elapsed = self.get_time(name)
        return 1.0 / elapsed if elapsed > 0 else 0
    
    def report(self) -> Dict[str, float]:
        """获取性能报告"""
        report = {}
        for name, timing in self.timings.items():
            if timing.get("end"):
                report[name] = timing["end"] - timing["start"]
        return report


# 全局实例
_optimizer = None
_cache = None
_timer = None

def get_optimizer() -> PerformanceOptimizer:
    """获取优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer()
    return _optimizer

def get_cache() -> ModelCache:
    """获取缓存实例"""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache

def get_timer() -> InferenceTimer:
    """获取计时器"""
    global _timer
    if _timer is None:
        _timer = InferenceTimer()
    return _timer


# 性能配置
PERFORMANCE_CONFIG = {
    "enable_cache": True,
    "enable_warmup": True,
    "default_batch_size": 4,
    "max_queue_size": 10,
    "timeout_seconds": 30,
}
