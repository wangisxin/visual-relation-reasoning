"""
批量处理模块
支持多图批量处理和结果导出
"""
import os
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime


class BatchProcessor:
    """
    批量处理器
    
    功能:
    - 多图批量检测
    - 结果汇总导出
    - 进度跟踪
    """
    
    def __init__(self, detector=None):
        self.detector = detector
        self.results = []
        
    def process_folder(self, folder_path: str, 
                     extensions: List[str] = ['.jpg', '.jpeg', '.png'],
                     conf_threshold: float = 0.25) -> List[Dict]:
        """
        处理文件夹中的所有图片
        
        Args:
            folder_path: 文件夹路径
            extensions: 支持的图片格式
            conf_threshold: 置信度阈值
            
        Returns:
            list: 所有图片的检测结果
        """
        results = []
        image_files = []
        
        # 获取所有图片文件
        for ext in extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        total = len(image_files)
        progress_bar = st.progress(0)
        
        for i, img_path in enumerate(image_files):
            try:
                # 检测
                detections = self.detector.detect(str(img_path), conf_threshold)
                
                results.append({
                    'file': str(img_path.name),
                    'path': str(img_path),
                    'detections': detections,
                    'count': len(detections),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'file': str(img_path.name),
                    'path': str(img_path),
                    'detections': [],
                    'count': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e)
                })
            
            # 更新进度
            progress_bar.progress((i + 1) / total)
        
        progress_bar.empty()
        self.results = results
        return results
    
    def process_file_list(self, file_paths: List[str],
                         conf_threshold: float = 0.25) -> List[Dict]:
        """
        处理文件列表
        
        Args:
            file_paths: 文件路径列表
            conf_threshold: 置信度阈值
            
        Returns:
            list: 检测结果
        """
        results = []
        total = len(file_paths)
        progress_bar = st.progress(0)
        
        for i, img_path in enumerate(file_paths):
            try:
                detections = self.detector.detect(img_path, conf_threshold)
                
                results.append({
                    'file': os.path.basename(img_path),
                    'path': img_path,
                    'detections': detections,
                    'count': len(detections),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'file': os.path.basename(img_path),
                    'path': img_path,
                    'detections': [],
                    'count': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e)
                })
            
            progress_bar.progress((i + 1) / total)
        
        progress_bar.empty()
        self.results = results
        return results
    
    def export_json(self, output_path: str):
        """导出JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def export_csv(self, output_path: str):
        """导出CSV格式"""
        rows = []
        for r in self.results:
            row = {
                'file': r['file'],
                'count': r['count'],
                'status': r['status'],
                'timestamp': r['timestamp']
            }
            
            # 添加检测到的类别
            classes = [d['class_name'] for d in r.get('detections', [])]
            row['classes'] = ', '.join(classes)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def export_excel(self, output_path: str):
        """导出Excel格式"""
        rows = []
        for r in self.results:
            row = {
                '文件名': r['file'],
                '检测数量': r['count'],
                '状态': r['status'],
                '时间': r['timestamp']
            }
            
            # 添加检测到的类别
            classes = [d['class_name'] for d in r.get('detections', [])]
            row['检测类别'] = ', '.join(classes)
            
            # 添加每个类别的数量
            from collections import Counter
            class_counts = Counter([d['class_name'] for d in r.get('detections', [])])
            for cls, count in class_counts.items():
                row[f'{cls}数量'] = count
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取汇总统计"""
        if not self.results:
            return {}
        
        total_files = len(self.results)
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = total_files - success_count
        
        # 统计所有检测到的类别
        all_classes = []
        for r in self.results:
            all_classes.extend([d['class_name'] for d in r.get('detections', [])])
        
        from collections import Counter
        class_counts = Counter(all_classes)
        
        return {
            'total_files': total_files,
            'success_count': success_count,
            'error_count': error_count,
            'total_detections': len(all_classes),
            'class_counts': dict(class_counts.most_common(10)),
            'success_rate': success_count / total_files if total_files > 0 else 0
        }


def batch_process_ui():
    """批量处理UI"""
    st.header("📂 批量处理")
    
    # 选择处理方式
    process_type = st.radio("处理方式", ["文件夹", "文件列表"])
    
    processor = BatchProcessor()
    
    if process_type == "文件夹":
        folder_path = st.text_input("文件夹路径")
        if st.button("开始处理"):
            if folder_path and os.path.isdir(folder_path):
                with st.spinner("处理中..."):
                    results = processor.process_folder(folder_path)
                    
                    # 显示汇总
                    summary = processor.get_summary()
                    st.success(f"处理完成! 成功: {summary['success_count']}, 失败: {summary['error_count']}")
                    
                    # 显示统计
                    if summary.get('class_counts'):
                        st.write("检测类别统计:")
                        for cls, count in summary['class_counts'].items():
                            st.write(f"- {cls}: {count}")
            else:
                st.error("请输入有效的文件夹路径")
    
    else:
        uploaded_files = st.file_uploader("选择图片", 
                                         type=['jpg', 'jpeg', 'png'],
                                         accept_multiple_files=True)
        if st.button("开始处理") and uploaded_files:
            with st.spinner("处理中..."):
                file_paths = [f.name for f in uploaded_files]
                results = processor.process_file_list(file_paths)
                
                summary = processor.get_summary()
                st.success(f"处理完成!")
    
    # 导出选项
    if processor.results:
        st.divider()
        st.subheader("💾 导出结果")
        
        col1, col2, col3 = st.columns(3)
        
        if col1.button("导出JSON"):
            processor.export_json("batch_results.json")
            st.success("已导出JSON")
        
        if col2.button("导出CSV"):
            processor.export_csv("batch_results.csv")
            st.success("已导出CSV")
        
        if col3.button("导出Excel"):
            processor.export_excel("batch_results.xlsx")
            st.success("已导出Excel")
