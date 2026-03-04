# 主题配置
THEMES = {
    "light": {
        "name": "浅色主题",
        "primary_color": "#3B82F6",
        "background": "#FFFFFF",
        "secondary_background": "#F3F4F6",
        "text_color": "#1F2937",
        "font": "sans-serif"
    },
    "dark": {
        "name": "深色主题", 
        "primary_color": "#60A5FA",
        "background": "#1F2937",
        "secondary_background": "#374151",
        "text_color": "#F9FAFB",
        "font": "sans-serif"
    },
    "blue": {
        "name": "蓝色主题",
        "primary_color": "#0EA5E9",
        "background": "#F0F9FF",
        "secondary_background": "#E0F2FE",
        "text_color": "#0C4A6E",
        "font": "sans-serif"
    },
    "green": {
        "name": "绿色主题",
        "primary_color": "#10B981",
        "background": "#ECFDF5",
        "secondary_background": "#D1FAE5",
        "text_color": "#064E3B",
        "font": "sans-serif"
    }
}

# 应用主题
def apply_theme(theme_name: str = "light"):
    """应用主题样式"""
    theme = THEMES.get(theme_name, THEMES["light"])
    
    st.markdown(f"""
    <style>
    :root {{
        --primary-color: {theme['primary_color']};
        --background: {theme['background']};
        --secondary-background: {theme['secondary_background']};
        --text-color: {theme['text_color']};
    }}
    
    .stApp {{
        background-color: {theme['background']};
        color: {theme['text_color']};
    }}
    
    .stButton > button {{
        background-color: {theme['primary_color']};
        color: white;
        border-radius: 8px;
    }}
    
    .stSidebar {{
        background-color: {theme['secondary_background']};
    }}
    
    h1, h2, h3 {{
        color: {theme['primary_color']};
    }}
    </style>
    """, unsafe_allow_html=True)


# 动画效果
ANIMATIONS = """
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}
</style>
"""


# 加载动画
def loading_spinner(text: str = "加载中..."):
    """加载动画"""
    return st.spinner(f"⏳ {text}")


# 成功/错误提示
def show_success(message: str):
    """显示成功消息"""
    st.success(f"✅ {message}")


def show_error(message: str):
    """显示错误消息"""
    st.error(f"❌ {message}")


def show_warning(message: str):
    """显示警告消息"""
    st.warning(f"⚠️ {message}")


def show_info(message: str):
    """显示信息消息"""
    st.info(f"ℹ️ {message}")
