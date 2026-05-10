# LLM API config (OpenAI-compatible endpoint)
# 复制此文件为 config.py 并填入你的 API Key
# config.py 已被 .gitignore 忽略，不会上传到 GitHub

LLM_CONFIG = {
    "api_key": "your-api-key-here",
    "model_name": "deepseek-v4-pro",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# Mem0 API config (Long-term memory)
# 未配置时系统优雅降级，使用短期记忆继续工作
MEM0_CONFIG = {
    "api_key": "your-mem0-key-here",
}
