"""
PostgreSQL 长期记忆 — 替代 Mem0 云服务

6 张表：
  students          — 学生画像核心数据
  knowledge_nodes   — 知识点掌握度分布
  error_records     — 错误模式分类轨迹
  session_snapshots — 会话历史快照
  review_queue      — 艾宾浩斯复习调度
  session_summaries — 会话总结 + pgvector 向量相似搜索

API 与当前 LongTermMemory 完全兼容。
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    import asyncpg
    ASYNC_PG_AVAILABLE = True
except ImportError:
    ASYNC_PG_AVAILABLE = False
    logger.warning("asyncpg not installed. PostgreSQL long-term memory disabled.")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class PostgresLongTermMemory:
    """PostgreSQL 长期记忆管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not ASYNC_PG_AVAILABLE:
            self.enabled = False
            logger.warning("asyncpg not available. PostgreSQL long-term memory disabled.")
            return

        self.config = config or {}
        self.enabled = False

        # 连接参数
        self.dsn = self.config.get(
            "dsn",
            "postgresql://localhost:5432/edux_memory"
        )
        self.embedding_dim = self.config.get("embedding_dim", 512)

        # Embedding 模型（懒加载）
        self._embedding_model = None
        self._pool = None

        # 标记需要异步初始化
        self._initialized = False
        logger.info("PostgresLongTermMemory configured (call await .initialize() to connect)")

    async def initialize(self):
        """异步初始化：连接池 + 建表 + 加载 embedding 模型"""
        if self._initialized:
            return

        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=2,
                max_size=10,
            )
            await self._ensure_tables()
            self._ensure_embedding_model()
            self.enabled = True
            self._initialized = True
            logger.info("PostgresLongTermMemory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL long-term memory: {e}")
            self.enabled = False

    def _ensure_embedding_model(self):
        if not EMBEDDING_AVAILABLE:
            logger.warning("sentence-transformers not available, vector search disabled")
            return
        if self._embedding_model is not None:
            return
        model_name = self.config.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        logger.info(f"Loading embedding model: {model_name}")
        self._embedding_model = SentenceTransformer(model_name, device="cpu")
        actual_dim = self._embedding_model.get_sentence_embedding_dimension()
        if actual_dim != self.embedding_dim:
            logger.warning(f"Embedding dim mismatch: configured {self.embedding_dim}, model has {actual_dim}")
            self.embedding_dim = actual_dim
        logger.info(f"Embedding model loaded (dim={self.embedding_dim})")

    async def _ensure_tables(self):
        """建表（幂等）"""
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(128),
                    grade_level VARCHAR(32),
                    learning_style VARCHAR(32) DEFAULT 'unknown',
                    style_confidence FLOAT DEFAULT 0.0,
                    preferred_pace VARCHAR(16) DEFAULT 'normal',
                    avg_attention_span INT DEFAULT 15,
                    optimal_session_length INT DEFAULT 25,
                    total_interactions INT DEFAULT 0,
                    extra JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(64) REFERENCES students(student_id) ON DELETE CASCADE,
                    kp_id VARCHAR(128) NOT NULL,
                    name VARCHAR(256) NOT NULL,
                    category VARCHAR(64) DEFAULT 'unknown',
                    mastery VARCHAR(32) DEFAULT 'not_started',
                    confidence FLOAT DEFAULT 0.0,
                    interaction_count INT DEFAULT 0,
                    error_count INT DEFAULT 0,
                    last_reviewed TIMESTAMP,
                    next_review_due TIMESTAMP,
                    prerequisites JSONB DEFAULT '[]',
                    UNIQUE(student_id, kp_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS error_records (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(64) REFERENCES students(student_id) ON DELETE CASCADE,
                    kp_id VARCHAR(128) NOT NULL,
                    error_type VARCHAR(64) NOT NULL,
                    question_snippet TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS session_snapshots (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(64) REFERENCES students(student_id) ON DELETE CASCADE,
                    session_id VARCHAR(128) NOT NULL,
                    topics_touched JSONB DEFAULT '[]',
                    questions_asked INT DEFAULT 0,
                    correct_ratio FLOAT,
                    detected_style VARCHAR(32),
                    difficulty_feedback VARCHAR(32),
                    mood VARCHAR(32),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(64) REFERENCES students(student_id) ON DELETE CASCADE,
                    kp_id VARCHAR(128) NOT NULL,
                    due_date TIMESTAMP NOT NULL,
                    mastery_at_schedule VARCHAR(32),
                    completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    student_id VARCHAR(64),
                    question TEXT NOT NULL,
                    answer TEXT,
                    summary_text TEXT,
                    mode VARCHAR(32),
                    agents_count INT,
                    subtasks_count INT,
                    total_time FLOAT,
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({self.embedding_dim}),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 向量索引（幂等）
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_session_summaries_embedding
                    ON session_summaries USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except Exception:
                pass  # 空表时创建 ivfflat 索引可能失败

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_summaries_student
                ON session_summaries(student_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_student
                ON knowledge_nodes(student_id)
            """)

            logger.info("Database tables ensured")

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self._embedding_model:
            return None
        try:
            vec = self._embedding_model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    # ===== 与当前 LongTermMemory 兼容的 API =====

    async def search_similar_sessions(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索历史会话"""
        if not self.enabled or not self._embedding_model:
            return []

        embedding = self._embed(query)
        if not embedding:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT session_id, question, answer, summary_text,
                           mode, metadata, created_at,
                           1 - (embedding <=> $1::vector) AS score
                    FROM session_summaries
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(embedding),
                    limit,
                )

            # 去重（基于内容哈希）
            seen = set()
            results = []
            for row in rows:
                content_hash = hashlib.md5(
                    f"{row['question']}:{row['answer'] or ''}".encode()
                ).hexdigest()
                if content_hash in seen:
                    continue
                seen.add(content_hash)

                results.append({
                    "memory_id": str(row["session_id"]),
                    "content": f"问题：{row['question']}\n回答：{(row['answer'] or '')[:500]}",
                    "score": round(row["score"], 4),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "timestamp": row["created_at"].isoformat() if row["created_at"] else "",
                })

            logger.info(f"Found {len(results)} similar sessions")
            return results
        except Exception as e:
            logger.error(f"Similar session search failed: {e}")
            return []

    async def add_session_summary(
        self,
        session_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """存储会话总结到 PostgreSQL"""
        if not self.enabled:
            return None

        embedding = self._embed(question) if self._embedding_model else None
        summary_text = f"问题：{question}\n回答：{(answer or '')[:500]}"

        student_id = None
        if metadata:
            student_id = metadata.get("student_id")

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO session_summaries
                        (session_id, student_id, question, answer, summary_text,
                         mode, agents_count, subtasks_count, total_time,
                         metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::vector)
                    """,
                    session_id,
                    student_id,
                    question,
                    answer,
                    summary_text,
                    (metadata or {}).get("mode", ""),
                    (metadata or {}).get("agents_count", 0),
                    (metadata or {}).get("subtasks_count", 0),
                    (metadata or {}).get("total_time", 0),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    str(embedding) if embedding else None,
                )

            logger.info(f"Saved session summary: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")
            return None

    # ===== StudentProfile CRUD（替代原来裸调 .mem0 的方式）=====

    async def get_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """从 PostgreSQL 加载完整学生画像"""
        if not self.enabled:
            return None

        try:
            async with self._pool.acquire() as conn:
                # 加载学生基础数据
                student_row = await conn.fetchrow(
                    "SELECT * FROM students WHERE student_id = $1", student_id
                )
                if not student_row:
                    return None

                # 加载知识节点
                kn_rows = await conn.fetch(
                    "SELECT * FROM knowledge_nodes WHERE student_id = $1", student_id
                )
                knowledge_graph = {}
                for row in kn_rows:
                    knowledge_graph[row["kp_id"]] = {
                        "id": row["kp_id"],
                        "name": row["name"],
                        "category": row["category"],
                        "parent_id": None,
                        "prerequisites": json.loads(row["prerequisites"]) if row["prerequisites"] else [],
                        "mastery": row["mastery"],
                        "confidence": row["confidence"],
                        "interaction_count": row["interaction_count"],
                        "last_reviewed": row["last_reviewed"].isoformat() if row["last_reviewed"] else None,
                        "next_review_due": row["next_review_due"].isoformat() if row["next_review_due"] else None,
                        "error_count": row["error_count"],
                    }

                # 加载错误记录
                err_rows = await conn.fetch(
                    "SELECT * FROM error_records WHERE student_id = $1 ORDER BY created_at DESC LIMIT 200",
                    student_id,
                )
                error_log = [
                    {
                        "knowledge_point_id": r["kp_id"],
                        "error_type": r["error_type"],
                        "question_snippet": r["question_snippet"] or "",
                        "timestamp": r["created_at"].isoformat(),
                        "resolved": r["resolved"],
                    }
                    for r in err_rows
                ]

                # 加载会话历史
                snap_rows = await conn.fetch(
                    "SELECT * FROM session_snapshots WHERE student_id = $1 ORDER BY created_at DESC LIMIT 50",
                    student_id,
                )
                session_history = [
                    {
                        "session_id": r["session_id"],
                        "timestamp": r["created_at"].isoformat(),
                        "topics_touched": json.loads(r["topics_touched"]) if r["topics_touched"] else [],
                        "questions_asked": r["questions_asked"],
                        "correct_ratio": r["correct_ratio"],
                        "detected_style": r["detected_style"],
                        "difficulty_feedback": r["difficulty_feedback"],
                        "mood": r["mood"],
                    }
                    for r in snap_rows
                ]

                # 加载复习队列
                rq_rows = await conn.fetch(
                    "SELECT * FROM review_queue WHERE student_id = $1 AND completed = FALSE ORDER BY due_date",
                    student_id,
                )
                review_queue = [
                    {
                        "kp_id": r["kp_id"],
                        "due_date": r["due_date"].isoformat(),
                        "mastery_at_schedule": r["mastery_at_schedule"],
                    }
                    for r in rq_rows
                ]

                extra = json.loads(student_row["extra"]) if student_row["extra"] else {}

                return {
                    "student_id": student_row["student_id"],
                    "name": student_row["name"],
                    "grade_level": student_row["grade_level"],
                    "learning_style": student_row["learning_style"],
                    "style_confidence": student_row["style_confidence"],
                    "preferred_pace": student_row["preferred_pace"],
                    "avg_attention_span": student_row["avg_attention_span"],
                    "optimal_session_length": student_row["optimal_session_length"],
                    "total_interactions": student_row["total_interactions"],
                    "knowledge_graph": knowledge_graph,
                    "error_log": error_log,
                    "session_history": session_history,
                    "review_queue": review_queue,
                    "created_at": student_row["created_at"].isoformat(),
                    "updated_at": student_row["updated_at"].isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to load student profile {student_id}: {e}")
            return None

    async def save_student_profile(self, student_id: str, data: Dict[str, Any]) -> bool:
        """保存完整学生画像到 PostgreSQL（UPSERT 语义）"""
        if not self.enabled:
            return False

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # 1. UPSERT 学生基础数据
                    await conn.execute(
                        """
                        INSERT INTO students
                            (student_id, name, grade_level, learning_style,
                             style_confidence, preferred_pace, avg_attention_span,
                             optimal_session_length, total_interactions, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                        ON CONFLICT (student_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            grade_level = EXCLUDED.grade_level,
                            learning_style = EXCLUDED.learning_style,
                            style_confidence = EXCLUDED.style_confidence,
                            preferred_pace = EXCLUDED.preferred_pace,
                            avg_attention_span = EXCLUDED.avg_attention_span,
                            optimal_session_length = EXCLUDED.optimal_session_length,
                            total_interactions = EXCLUDED.total_interactions,
                            updated_at = NOW()
                        """,
                        student_id,
                        data.get("name"),
                        data.get("grade_level"),
                        data.get("learning_style", "unknown"),
                        data.get("style_confidence", 0.0),
                        data.get("preferred_pace", "normal"),
                        data.get("avg_attention_span", 15),
                        data.get("optimal_session_length", 25),
                        data.get("total_interactions", 0),
                    )

                    # 2. 知识节点：先删后插（简单方案）
                    await conn.execute(
                        "DELETE FROM knowledge_nodes WHERE student_id = $1", student_id
                    )
                    for kp_id, node in data.get("knowledge_graph", {}).items():
                        await conn.execute(
                            """
                            INSERT INTO knowledge_nodes
                                (student_id, kp_id, name, category, mastery, confidence,
                                 interaction_count, error_count, last_reviewed,
                                 next_review_due, prerequisites)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            """,
                            student_id,
                            kp_id,
                            node.get("name", kp_id),
                            node.get("category", "unknown"),
                            node.get("mastery", "not_started"),
                            node.get("confidence", 0.0),
                            node.get("interaction_count", 0),
                            node.get("error_count", 0),
                            _parse_timestamp(node.get("last_reviewed")),
                            _parse_timestamp(node.get("next_review_due")),
                            json.dumps(node.get("prerequisites", [])),
                        )

                    # 3. 错误记录：追加新记录（不删旧的）
                    for err in data.get("error_log", []):
                        await conn.execute(
                            """
                            INSERT INTO error_records
                                (student_id, kp_id, error_type, question_snippet, resolved)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT DO NOTHING
                            """,
                            student_id,
                            err.get("knowledge_point_id", ""),
                            err.get("error_type", ""),
                            err.get("question_snippet", ""),
                            err.get("resolved", False),
                        )

                    # 4. 复习队列：先删未完成的，再插
                    await conn.execute(
                        "DELETE FROM review_queue WHERE student_id = $1 AND completed = FALSE",
                        student_id,
                    )
                    for item in data.get("review_queue", []):
                        await conn.execute(
                            """
                            INSERT INTO review_queue
                                (student_id, kp_id, due_date, mastery_at_schedule)
                            VALUES ($1, $2, $3, $4)
                            """,
                            student_id,
                            item.get("kp_id", ""),
                            _parse_timestamp(item.get("due_date")) or datetime.now(),
                            item.get("mastery_at_schedule", ""),
                        )

                    # 5. 会话快照：只插最新的（避免重复）
                    for snap in data.get("session_history", []):
                        sid = snap.get("session_id", "")
                        if not sid:
                            continue
                        existing = await conn.fetchval(
                            "SELECT 1 FROM session_snapshots WHERE session_id = $1", sid
                        )
                        if existing:
                            continue
                        await conn.execute(
                            """
                            INSERT INTO session_snapshots
                                (student_id, session_id, topics_touched, questions_asked,
                                 correct_ratio, detected_style, difficulty_feedback, mood)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            student_id,
                            sid,
                            json.dumps(snap.get("topics_touched", [])),
                            snap.get("questions_asked", 0),
                            snap.get("correct_ratio"),
                            snap.get("detected_style"),
                            snap.get("difficulty_feedback"),
                            snap.get("mood"),
                        )

            logger.info(f"Saved student profile: {student_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save student profile {student_id}: {e}")
            return False

    async def close(self):
        if self._pool:
            await self._pool.close()
            logger.info("PostgresLongTermMemory connection pool closed")


def _parse_timestamp(val) -> Optional[datetime]:
    """安全解析时间戳"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None
