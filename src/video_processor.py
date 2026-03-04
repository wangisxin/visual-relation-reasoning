"""
视频流处理模块
支持摄像头和视频文件的实时检测
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import streamlit as st
from dataclasses import dataclass


@dataclass
class VideoConfig:
    """视频配置"""
    width: int = 1280
    height: int = 720
    fps: int = 30
    codec: str = 'mp4v'


class VideoProcessor:
    """
    视频流处理器
    
    支持:
    - 摄像头实时输入
    - 视频文件处理
    - 帧处理回调
    """
    
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.cap = None
        self.writer = None
        self.is_opened = False
        
    def open_camera(self, camera_id: int = 0) -> bool:
        """
        打开摄像头
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            bool: 是否成功
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.is_opened = self.cap.isOpened()
        
        if self.is_opened:
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        return self.is_opened
    
    def open_video(self, video_path: str) -> bool:
        """
        打开视频文件
        
        Args:
            video_path: 视频路径
            
        Returns:
            bool: 是否成功
        """
        self.cap = cv2.VideoCapture(video_path)
        self.is_opened = self.cap.isOpened()
        
        if self.is_opened:
            # 获取视频属性
            self.config.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.config.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.config.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        return self.is_opened
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取下一帧
        
        Returns:
            (success, frame)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def write_frame(self, frame: np.ndarray):
        """写入帧"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def create_writer(self, output_path: str):
        """
        创建视频写入器
        
        Args:
            output_path: 输出路径
        """
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, self.config.fps,
            (self.config.width, self.config.height)
        )
    
    def process_video(self, process_func: Callable, 
                     output_path: Optional[str] = None,
                     show_progress: bool = True) -> list:
        """
        处理整个视频
        
        Args:
            process_func: 处理函数 (frame) -> annotated_frame
            output_path: 输出路径 (可选)
            show_progress: 是否显示进度
            
        Returns:
            list: 所有帧的检测结果
        """
        if not self.is_opened:
            return []
        
        # 创建写入器
        if output_path:
            self.create_writer(output_path)
        
        results = []
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0) if show_progress else None
        
        frame_idx = 0
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            
            # 处理帧
            annotated = process_func(frame)
            results.append(annotated)
            
            # 写入
            if self.writer:
                self.write_frame(annotated)
            
            # 更新进度
            if progress_bar:
                progress_bar.progress((frame_idx + 1) / frame_count)
            
            frame_idx += 1
        
        # 清理
        if progress_bar:
            progress_bar.empty()
        
        return results
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        self.is_opened = False


class StreamlitCamera:
    """
    Streamlit摄像头组件
    
    用于在Streamlit中显示摄像头输入
    """
    
    def __init__(self):
        self.frame_placeholder = None
        self.control_placeholder = None
    
    def create_ui(self):
        """创建UI组件"""
        self.frame_placeholder = st.empty()
        self.control_placeholder = st.empty()
    
    def display_frame(self, frame: np.ndarray):
        """显示帧"""
        if self.frame_placeholder:
            self.frame_placeholder.image(frame, channels="BGR")
    
    def display_controls(self, start_btn, stop_btn):
        """显示控制按钮"""
        if self.control_placeholder:
            col1, col2 = self.control_placeholder.columns(2)
            col1.button("开始", key=start_btn)
            col2.button("停止", key=stop_btn)


# 使用示例
def demo_video_processing():
    """演示视频处理"""
    processor = VideoProcessor()
    
    # 打开摄像头
    if processor.open_camera(0):
        st.success("摄像头已打开")
        
        # 处理帧
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            
            # TODO: 添加检测逻辑
            # annotated = detector.detect(frame)
            
            # 显示
            # cv2.imshow("Frame", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processor.release()
