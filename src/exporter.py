"""
结果导出模块
支持多种格式的检测结果导出
"""
import os
import json
import csv
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


class ResultExporter:
    """
    结果导出器
    
    支持格式:
    - JSON: 完整数据结构
    - CSV: 表格数据
    - Excel: 带统计图表
    - 图片: 带标注的图片
    - 报告: HTML/PDF报告
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_json(self, results: List[Dict], filename: str = None) -> str:
        """
        导出JSON格式
        
        Args:
            results: 检测结果列表
            filename: 文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 转换numpy类型为Python原生类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results_converted = convert(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def export_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        导出CSV格式
        
        Args:
            results: 检测结果列表
            filename: 文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        rows = []
        for r in results:
            row = {
                'file': r.get('file', ''),
                'count': r.get('count', 0),
                'status': r.get('status', 'unknown'),
                'timestamp': r.get('timestamp', '')
            }
            
            # 类别列表
            classes = [d['class_name'] for d in r.get('detections', [])]
            row['classes'] = ';'.join(classes)
            
            # 每个类别数量
            class_counts = {}
            for d in r.get('detections', []):
                cls = d['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in class_counts.items():
                row[f'cnt_{cls}'] = count
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def export_excel(self, results: List[Dict], filename: str = None) -> str:
        """
        导出Excel格式
        
        Args:
            results: 检测结果列表
            filename: 文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 汇总表
        summary_rows = []
        for r in results:
            summary_rows.append({
                '文件名': r.get('file', ''),
                '检测数量': r.get('count', 0),
                '状态': r.get('status', 'unknown'),
                '时间': r.get('timestamp', '')
            })
        
        # 详细表
        detail_rows = []
        for r in results:
            file = r.get('file', '')
            for d in r.get('detections', []):
                detail_rows.append({
                    '文件名': file,
                    '类别': d.get('class_name', ''),
                    '置信度': d.get('confidence', 0),
                    'bbox': d.get('bbox', '')
                })
        
        # 写入Excel
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='汇总', index=False)
            if detail_rows:
                pd.DataFrame(detail_rows).to_excel(writer, sheet_name='详细', index=False)
        
        return output_path
    
    def export_images(self, results: List[Dict], 
                     image_folder: str = None,
                     filename: str = None) -> str:
        """
        导出标注图片
        
        Args:
            results: 检测结果列表
            image_folder: 原图文件夹
            filename: 文件名
            
        Returns:
            str: 输出文件夹路径
        """
        if filename is None:
            filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_folder = os.path.join(self.output_dir, filename)
        os.makedirs(output_folder, exist_ok=True)
        
        for r in results:
            if r.get('status') != 'success':
                continue
            
            img_path = r.get('path')
            if not img_path or not os.path.exists(img_path):
                continue
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 绘制标注
            for det in r.get('detections', []):
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{det.get('class_name', '')} {det.get('confidence', 0):.2f}"
                    cv2.putText(img, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存
            output_path = os.path.join(output_folder, r.get('file', 'output.jpg'))
            cv2.imwrite(output_path, img)
        
        return output_folder
    
    def export_html_report(self, results: List[Dict], 
                          filename: str = None) -> str:
        """
        导出HTML报告
        
        Args:
            results: 检测结果列表
            filename: 文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 统计
        total_files = len(results)
        success = sum(1 for r in results if r.get('status') == 'success')
        total_detections = sum(r.get('count', 0) for r in results)
        
        # 类别统计
        class_counts = {}
        for r in results:
            for d in r.get('detections', []):
                cls = d.get('class_name', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # 生成HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>检测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stat {{ display: inline-block; margin: 10px 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>🖼️ 图像检测报告</h1>
    
    <div class="summary">
        <h2>📊 统计汇总</h2>
        <div class="stat">📁 总文件数: <strong>{total_files}</strong></div>
        <div class="stat">✅ 成功: <strong>{success}</strong></div>
        <div class="stat">🔍 总检测数: <strong>{total_detections}</strong></div>
    </div>
    
    <h2>📈 类别统计</h2>
    <table>
        <tr><th>类别</th><th>数量</th></tr>
"""
        
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            html += f"        <tr><td>{cls}</td><td>{count}</td></tr>\n"
        
        html += """
    </table>
    
    <h2>📋 详细结果</h2>
    <table>
        <tr><th>文件名</th><th>检测数</th><th>状态</th><th>类别</th></tr>
"""
        
        for r in results:
            classes = ','.join([d.get('class_name', '') for d in r.get('detections', [])])
            html += f"""        <tr>
            <td>{r.get('file', '')}</td>
            <td>{r.get('count', 0)}</td>
            <td>{r.get('status', '')}</td>
            <td>{classes}</td>
        </tr>\n"""
        
        html += """
    </table>
    
    <footer>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </footer>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path


# 便捷函数
def quick_export(results: List[Dict], output_dir: str = "output",
                formats: List[str] = None) -> Dict[str, str]:
    """
    快速导出多种格式
    
    Args:
        results: 检测结果
        output_dir: 输出目录
        formats: 导出格式列表 ['json', 'csv', 'excel', 'html']
        
    Returns:
        dict: {格式: 文件路径}
    """
    if formats is None:
        formats = ['json', 'csv']
    
    exporter = ResultExporter(output_dir)
    outputs = {}
    
    if 'json' in formats:
        outputs['json'] = exporter.export_json(results)
    if 'csv' in formats:
        outputs['csv'] = exporter.export_csv(results)
    if 'excel' in formats:
        outputs['excel'] = exporter.export_excel(results)
    if 'html' in formats:
        outputs['html'] = exporter.export_html_report(results)
    
    return outputs
