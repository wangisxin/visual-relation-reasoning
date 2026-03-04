"""
配置管理模块
统一管理系统配置
"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import streamlit as st


@dataclass
class DetectorConfig:
    """检测器配置"""
    model_size: str = 'n'  # n/s/m/l/x
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = 'cpu'  # cpu/cuda/mps
    max_det: int = 300
    half_precision: bool = False


@dataclass
class VLMConfig:
    """VLM配置"""
    mode: str = 'lightweight'  # api/local/lightweight
    provider: str = 'openai'  # openai/anthropic/qwen
    model: str = 'gpt-4o'
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class SAMConfig:
    """SAM配置"""
    mode: str = 'lightweight'  # sam/grounded/lightweight
    model_type: str = 'vit_b'  # vit_b/vit_l/vit_h
    model_path: Optional[str] = None
    device: str = 'cpu'


@dataclass
class UIConfig:
    """UI配置"""
    theme: str = 'light'
    page_title: str = '多维图像识别与理解'
    show_fps: bool = True
    show_confidence: bool = True
    show_labels: bool = True
    default_tab: str = 'detection'


@dataclass
class AppConfig:
    """应用配置"""
    detector: DetectorConfig = None
    vlm: VLMConfig = None
    sam: SAMConfig = None
    ui: UIConfig = None
    
    def __post_init__(self):
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.vlm is None:
            self.vlm = VLMConfig()
        if self.sam is None:
            self.sam = SAMConfig()
        if self.ui is None:
            self.ui = UIConfig()


class ConfigManager:
    """
    配置管理器
    
    功能:
    - 配置加载/保存
    - 默认配置
    - 配置验证
    - Streamlit UI
    """
    
    DEFAULT_CONFIG = {
        'detector': {
            'model_size': 'n',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'max_det': 300,
            'half_precision': False
        },
        'vlm': {
            'mode': 'lightweight',
            'provider': 'openai',
            'model': 'gpt-4o',
            'temperature': 0.7,
            'max_tokens': 500
        },
        'sam': {
            'mode': 'lightweight',
            'model_type': 'vit_b',
            'device': 'cpu'
        },
        'ui': {
            'theme': 'light',
            'show_fps': True,
            'show_confidence': True,
            'show_labels': True
        }
    }
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_path()
        self.config = self.load()
    
    def _get_default_path(self) -> str:
        """获取默认配置路径"""
        home = os.path.expanduser('~')
        config_dir = os.path.join(home, '.config', 'image_reasoning')
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, 'config.json')
    
    def load(self) -> Dict:
        """加载配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return self.DEFAULT_CONFIG.copy()
    
    def save(self, config: Dict = None):
        """保存配置"""
        config = config or self.config
        
        # 添加元数据
        config['_meta'] = {
            'version': '1.0',
            'updated': datetime.now().isoformat()
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def reset(self):
        """重置为默认配置"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
    
    def render_ui(self):
        """渲染配置UI"""
        st.sidebar.header("⚙️ 配置管理")
        
        # 检测器配置
        with st.sidebar.expander("🎯 检测器"):
            model_size = st.selectbox(
                "模型大小",
                ['n', 's', 'm', 'l', 'x'],
                index=['n', 's', 'm', 'l', 'x'].index(
                    self.config.get('detector', {}).get('model_size', 'n')
                )
            )
            conf = st.slider("置信度", 0.1, 1.0, 
                           self.config.get('detector', {}).get('conf_threshold', 0.25))
            device = st.selectbox("设备", ['cpu', 'cuda', 'mps'],
                               index=['cpu', 'cuda', 'mps'].index(
                                   self.config.get('detector', {}).get('device', 'cpu')
                               ))
            
            self.config['detector'] = {
                'model_size': model_size,
                'conf_threshold': conf,
                'device': device
            }
        
        # VLM配置
        with st.sidebar.expander("🤖 VLM"):
            vlm_mode = st.selectbox("模式",
                                   ['lightweight', 'api', 'local'],
                                   index=['lightweight', 'api', 'local'].index(
                                       self.config.get('vlm', {}).get('mode', 'lightweight')
                                   ))
            provider = st.selectbox("提供商",
                                  ['openai', 'qwen', 'anthropic'],
                                  index=['openai', 'qwen', 'anthropic'].index(
                                      self.config.get('vlm', {}).get('provider', 'openai')
                                  ))
            
            self.config['vlm'] = {
                'mode': vlm_mode,
                'provider': provider
            }
        
        # UI配置
        with st.sidebar.expander("🎨 UI"):
            show_fps = st.checkbox("显示FPS",
                                  self.config.get('ui', {}).get('show_fps', True))
            show_labels = st.checkbox("显示标签",
                                      self.config.get('ui', {}).get('show_labels', True))
            
            self.config['ui'] = {
                'show_fps': show_fps,
                'show_labels': show_labels
            }
        
        # 保存按钮
        col1, col2 = st.sidebar.columns(2)
        if col1.button("💾 保存配置"):
            self.save()
            st.sidebar.success("配置已保存!")
        
        if col2.button("🔄 重置"):
            self.reset()
            st.sidebar.info("配置已重置")


# 全局配置实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
