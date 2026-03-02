"""
图像关系推理桌面应用
"""
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QTextEdit, QScrollArea, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from detector import ObjectDetector
from relation_reasoner import RelationReasoner


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.detector = None
        self.reasoner = RelationReasoner()
        self.current_image_path = None
        self.current_detections = None
        self.current_relations = None
        
        self.init_ui()
        self.init_detector()
    
    def init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("图像关系推理系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：图片显示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 图片标签
        self.image_label = QLabel("请上传图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 400px;")
        left_layout.addWidget(self.image_label)
        
        # 上传按钮
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_btn)
        
        # 保存结果按钮
        self.save_btn = QPushButton("保存结果图片")
        self.save_btn.clicked.connect(self.save_result_image)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)
        
        # 右侧：分析结果
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 结果标题
        result_label = QLabel("分析结果")
        result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(result_label)
        
        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-size: 12px;")
        right_layout.addWidget(self.result_text)
        
        # 添加到主布局
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)
    
    def init_detector(self):
        """初始化检测器"""
        try:
            self.detector = ObjectDetector('n')
            self.append_result("系统初始化完成，YOLOv8n 模型已加载\n")
        except Exception as e:
            self.append_result(f"⚠️ 警告: 模型自动加载失败\n")
            self.append_result(f"   错误: {e}\n")
            self.append_result("   首次上传图片时将自动下载模型\n")
            self.detector = None  # 延迟加载
    
    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.process_image(file_path)
    
    def process_image(self, image_path: str):
        """处理图片"""
        # 延迟加载检测器（如果还没加载）
        if self.detector is None:
            try:
                self.detector = ObjectDetector('n')
                self.append_result("模型加载成功\n")
            except Exception as e:
                self.append_result(f"❌ 模型加载失败: {e}\n")
                return
        
        self.append_result(f"\n{'='*50}")
        self.append_result(f"处理图片: {os.path.basename(image_path)}\n")
        
        # 显示原图
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        try:
            # 物体检测
            self.append_result("正在进行物体检测...\n")
            detections = self.detector.detect(image_path)
            self.current_detections = detections
            
            self.append_result(f"检测到 {len(detections)} 个物体:\n")
            for det in detections:
                self.append_result(f"  - {det['class_name']}: {det['confidence']:.2f}\n")
            
            # 关系推理
            self.append_result("\n正在进行关系推理...\n")
            relations = self.reasoner.reason(detections)
            self.current_relations = relations
            
            # 显示空间关系
            if relations['spatial_relations']:
                self.append_result("\n【空间关系】\n")
                for rel in relations['spatial_relations']:
                    self.append_result(f"  • {rel['relation']}\n")
            
            # 显示语义关系
            if relations['semantic_relations']:
                self.append_result("\n【语义关系】\n")
                for rel in relations['semantic_relations']:
                    self.append_result(f"  • {rel['relation']}\n")
            
            # 显示计数
            if relations['counts']:
                self.append_result("\n【物体计数】\n")
                for name, count in relations['counts'].items():
                    self.append_result(f"  - {name}: {count}个\n")
            
            # 绘制标注图
            annotated = self.detector.draw_detections(image_path, detections)
            self.current_annotated_image = annotated
            
            # 显示标注图
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(annotated_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            self.append_result("\n✅ 分析完成！\n")
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            self.append_result(f"\n❌ 处理失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    def save_result_image(self):
        """保存结果图片"""
        if not hasattr(self, 'current_annotated_image'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", 
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_annotated_image)
            self.append_result(f"\n📷 结果图片已保存: {file_path}\n")
    
    def append_result(self, text: str):
        """追加结果文本"""
        self.result_text.append(text)
        # 滚动到底部
        self.result_text.verticalScrollBar().setValue(
            self.result_text.verticalScrollBar().maximum()
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
