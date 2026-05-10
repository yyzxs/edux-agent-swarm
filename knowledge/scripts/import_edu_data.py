"""
导入教育知识文档到 Milvus 知识库

文件名约定：{subject}_{topic}.txt
  subject: 学科 — math / physics / english / chemistry
  topic:   主题 — trigonometry / mechanics / grammar

也兼容平铺式命名：直接以主题命名
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from knowledge.milvus_kb import EduKnowledgeBase


# 学科→中文名映射
SUBJECT_MAP = {
    "math": "数学", "physics": "物理", "english": "英语",
    "chemistry": "化学", "biology": "生物", "history": "历史",
    "geography": "地理", "politics": "政治", "chinese": "语文",
}

# 主题→知识点类型映射
TYPE_HINTS = {
    "grammar": "grammar", "trigonometry": "formula",
    "mechanics": "theory", "formula": "formula",
    "vocabulary": "vocabulary", "reading": "reading",
}


def parse_filename(stem: str) -> dict:
    """
    解析文件名，提取元数据

    支持两种格式：
      math_trigonometry  → subject=math, topic=trigonometry
      english_grammar    → subject=english, topic=grammar
    """
    info = {"subject": "", "topic": stem, "doc_type": "general"}

    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0] in SUBJECT_MAP:
        info["subject"] = parts[0]
        info["topic"] = parts[1]
    else:
        # 尝试从第一个词推断学科
        first = stem.split("_")[0]
        if first in SUBJECT_MAP:
            info["subject"] = first
            info["topic"] = "_".join(stem.split("_")[1:])
        else:
            info["subject"] = first

    # 推断知识点类型
    for hint, doc_type in TYPE_HINTS.items():
        if hint in info["topic"].lower():
            info["doc_type"] = doc_type
            break

    return info


def load_documents_from_directory(doc_dir: Path) -> list:
    """从 documents 目录加载所有 txt 文件"""
    documents = []
    txt_files = sorted(doc_dir.glob("*.txt"))

    if not txt_files:
        logger.warning(f"No txt files found in {doc_dir}")
        return documents

    logger.info(f"Found {len(txt_files)} txt files")

    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding="utf-8")
            stem = txt_file.stem
            info = parse_filename(stem)

            subject_cn = SUBJECT_MAP.get(info["subject"], info["subject"])

            # 尝试从内容第一行提取标题
            first_line = content.strip().split("\n")[0].lstrip("# ").strip()
            title = first_line or info["topic"]

            doc = {
                "id": f"edu_{stem}",
                "content": content,
                "metadata": {
                    "type": info["doc_type"],
                    "subject": info["subject"],
                    "subject_cn": subject_cn,
                    "topic": info["topic"],
                    "title": title,
                    "filename": txt_file.name,
                    "source": f"EduX教育知识库 ({subject_cn})",
                },
            }

            documents.append(doc)
            logger.info(f"  Loaded: {txt_file.name} → subject={info['subject']}, topic={info['topic']}, type={info['doc_type']}")

        except Exception as e:
            logger.error(f"Error loading {txt_file.name}: {e}")

    return documents


def main():
    logger.info("=" * 70)
    logger.info("导入教育知识文档到 Milvus 知识库")
    logger.info("=" * 70)

    doc_dir = Path(__file__).parent.parent / "data" / "documents"

    if not doc_dir.exists():
        logger.error(f"Documents directory not found: {doc_dir}")
        return

    # 加载所有文档
    all_docs = load_documents_from_directory(doc_dir)

    if not all_docs:
        logger.error("No documents loaded.")
        return

    # 按学科统计
    by_subject = {}
    for doc in all_docs:
        s = doc["metadata"]["subject"]
        by_subject.setdefault(s, 0)
        by_subject[s] += 1

    logger.info(f"\nLoaded {len(all_docs)} documents:")
    for subject, count in sorted(by_subject.items()):
        logger.info(f"  {SUBJECT_MAP.get(subject, subject)}: {count}")

    # 导入到 Milvus
    logger.info("\nImporting to Milvus...")
    kb = EduKnowledgeBase()
    num_added = kb.add_documents(all_docs)

    logger.info("\n" + "=" * 70)
    logger.info(f"Done. {num_added} chunks imported.")
    logger.info("=" * 70)

    # 测试检索
    logger.info("\nTest search:")
    test_queries = ["三角函数公式", "力学牛顿定律", "英语虚拟语气"]

    for query in test_queries:
        results = kb.search(query, top_k=2)
        if results:
            best = results[0]
            logger.info(f"  '{query}' → {best['metadata'].get('title', 'N/A')} (score: {best['score']:.3f})")
        else:
            logger.warning(f"  '{query}' → no results")


if __name__ == "__main__":
    main()
