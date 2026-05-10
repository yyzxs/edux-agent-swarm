"""
EduX 教育知识库模块
"""
from .milvus_kb import EduKnowledgeBase

MilvusKB = EduKnowledgeBase  # alias for skill scripts

__all__ = ['EduKnowledgeBase', 'MilvusKB']
