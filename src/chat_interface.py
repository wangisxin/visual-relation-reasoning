"""
自然语言交互模块
支持通过自然语言与系统交互
"""
from typing import List, Dict, Optional, Tuple
import re
import streamlit as st


class NLUParser:
    """
    自然语言理解解析器
    
    功能:
    - 意图识别
    - 参数提取
    - 命令解析
    """
    
    # 意图定义
    INTENTS = {
        'detect': {
            'keywords': ['检测', '识别', '找出', '发现', 'detect', 'find'],
            'description': '检测图像中的物体'
        },
        'describe': {
            'keywords': ['描述', '说明', '介绍', 'describe', 'explain'],
            'description': '描述图像内容'
        },
        'analyze': {
            'keywords': ['分析', '解析', 'analyze', 'analysis'],
            'description': '分析图像'
        },
        'compare': {
            'keywords': ['比较', '对比', 'compare', 'vs'],
            'description': '比较物体'
        },
        'count': {
            'keywords': ['数', '多少', 'count', 'how many'],
            'description': '计数'
        },
        'locate': {
            'keywords': ['在哪', '位置', 'locate', 'where'],
            'description': '定位物体位置'
        },
        'segment': {
            'keywords': ['分割', '抠图', 'segment', 'cut'],
            'description': '分割物体'
        },
        'export': {
            'keywords': ['导出', '保存', 'export', 'save'],
            'description': '导出结果'
        },
        'help': {
            'keywords': ['帮助', 'help', '怎么', '如何'],
            'description': '获取帮助'
        },
        'setting': {
            'keywords': ['设置', '配置', 'setting', 'config'],
            'description': '系统设置'
        }
    }
    
    def __init__(self):
        self.context = {}
    
    def parse(self, text: str) -> Tuple[str, Dict]:
        """
        解析自然语言
        
        Args:
            text: 用户输入
            
        Returns:
            (intent, params): 意图和参数
        """
        text = text.lower().strip()
        
        # 意图识别
        intent = self._recognize_intent(text)
        
        # 参数提取
        params = self._extract_params(text, intent)
        
        return intent, params
    
    def _recognize_intent(self, text: str) -> str:
        """识别意图"""
        scores = {}
        
        for intent, info in self.INTENTS.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in text:
                    score += 1
            scores[intent] = score
        
        # 返回得分最高的意图
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                return best_intent
        
        return 'unknown'
    
    def _extract_params(self, text: str, intent: str) -> Dict:
        """提取参数"""
        params = {}
        
        # 提取置信度
        conf_match = re.search(r'(\d+%)', text)
        if conf_match:
            params['confidence'] = int(conf_match.group(1).replace('%', '')) / 100
        
        # 提取类别
        classes = ['person', 'car', 'dog', 'cat', 'chair', 'table', 'bottle', 'cup',
                  'phone', 'book', 'laptop', 'computer', 'tv']
        for cls in classes:
            if cls in text:
                params['class'] = cls
        
        # 提取数字
        num_match = re.search(r'(\d+)个?', text)
        if num_match:
            params['number'] = int(num_match.group(1))
        
        return params
    
    def generate_response(self, intent: str, result: any) -> str:
        """
        生成自然语言响应
        
        Args:
            intent: 意图
            result: 处理结果
            
        Returns:
            str: 自然语言响应
        """
        if intent == 'detect':
            count = len(result) if result else 0
            if count == 0:
                return "抱歉，我没有检测到任何物体。"
            
            classes = list(set([r.get('class_name', '') for r in result]))
            return f"我在图像中检测到了 {count} 个物体，包括: {', '.join(classes)}"
        
        elif intent == 'count':
            return f"图像中共有 {len(result) if result else 0} 个物体"
        
        elif intent == 'describe':
            return result if result else "我无法描述这个图像"
        
        elif intent == 'help':
            return """我可以帮你:
- 检测图像中的物体
- 描述图像内容
- 分析物体关系
- 分割特定物体
- 统计物体数量
- 导出检测结果

请告诉我你想做什么?"""
        
        else:
            return "我理解了你的请求，正在处理..."


class ChatInterface:
    """
    聊天界面
    
    功能:
    - 对话历史
    - 上下文管理
    - 多轮对话
    """
    
    def __init__(self):
        self.parser = NLUParser()
        self.history = []
        self.max_history = 20
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': None  # 可以添加时间戳
        })
        
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> List[Dict]:
        """获取历史"""
        return self.history
    
    def clear_history(self):
        """清空历史"""
        self.history = []
    
    def process(self, user_input: str, processor) -> str:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入
            processor: 处理器 (检测/VLM等)
            
        Returns:
            str: 系统响应
        """
        # 解析意图
        intent, params = self.parser.parse(user_input)
        
        # 添加用户消息
        self.add_message('user', user_input)
        
        # 处理请求
        response = self._handle_request(intent, params, user_input, processor)
        
        # 添加系统消息
        self.add_message('assistant', response)
        
        return response
    
    def _handle_request(self, intent: str, params: Dict, 
                      original_text: str, processor) -> str:
        """处理请求"""
        # 这里需要根据实际处理器来实现
        # 这是一个示例
        
        if intent == 'detect':
            # 执行检测
            # result = processor.detect(...)
            return "请上传图片，我会帮你检测其中的物体"
        
        elif intent == 'describe':
            # 生成描述
            # result = processor.describe(...)
            return "请上传图片，我会帮你描述其中的内容"
        
        elif intent == 'count':
            return "请上传图片，我会帮你统计其中的物体数量"
        
        elif intent == 'help':
            return self.parser.generate_response('help', None)
        
        else:
            return "我理解了你的请求，但需要更多信息。请上传图片或说明你想做什么。"
    
    def render(self):
        """渲染聊天界面"""
        st.subheader("💬 自然语言交互")
        
        # 显示历史
        for msg in self.history:
            if msg['role'] == 'user':
                st.markdown(f"**你:** {msg['content']}")
            else:
                st.markdown(f"**助手:** {msg['content']}")
        
        # 输入框
        user_input = st.text_input("请输入你的请求:", 
                                   key="chat_input",
                                   placeholder="例如: 检测这张图片中的物体")
        
        if st.button("发送"):
            if user_input:
                # 处理
                response = self.process(user_input, None)
                st.markdown(f"**助手:** {response}")
                st.rerun()
        
        if st.button("清空对话"):
            self.clear_history()
            st.rerun()


# 对话模板
CHAT_TEMPLATES = {
    'welcome': "你好!我可以帮你完成以下任务:\n\n" +
               "🔍 物体检测 - 识别图片中的物体\n" +
               "📝 图像描述 - 描述图片内容\n" +
               "📊 统计分析 - 统计物体数量\n" +
               "✂️ 物体分割 - 分割特定物体\n\n" +
               "请上传图片或告诉我你想做什么?",
    
    'detecting': "正在检测图片中的物体...",
    'done': "处理完成!",
    'error': "抱歉，处理过程中出现了一些问题。"
}


def create_chat_assistant() -> ChatInterface:
    """创建聊天助手"""
    return ChatInterface()
