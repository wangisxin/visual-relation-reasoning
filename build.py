"""
打包脚本 - 生成 Windows exe
"""
import PyInstaller.__main__
import os
import shutil

# 清理旧构建
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

PyInstaller.__main__.run([
    'src/main.py',
    '--name=图像关系推理',
    '--onefile',
    '--windowed',
    '--icon=icon.ico',  # 可选：添加图标
    '--add-data=src;src',
    '--hidden-import=ultralytics',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PyQt6',
    '--collect-all=ultralytics',
])
