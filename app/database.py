"""Task 5: database trail. One row per prediction request.

Works with SQLite (local runs) and PostgreSQL (Docker Compose) through SQLAlchemy;
only DATABASE_URL changes.
"""
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
                        desc, func, select, text)

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),   # when
    Column("filename", String(255), nullable=False),                  # sanitized name only
    Column("image_sha256", String(64), nullable=False),               # identifies the image without storing it
    Column("verdict", String(32), nullable=False),                    # what was decided
    Column("confidence", Float, nullable=False),                      # how confidently
    Column("prob_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(128), nullable=False),                # by which model
    Column("model_version", String(128), nullable=False),
    Column("inference_mode", String(16), nullable=False),             # local model or cloud API
    Column("latency_ms", Float),
)


class Store:
    def __init__(self, url: str):
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        metadata.create_all(self.engine)

    def add(self, filename: str, result: dict) -> int:
        row = {
            "created_at": datetime.now(timezone.utc),
            "filename": filename,
            "image_sha256": result["sha256"],
            "verdict": result["label"],
            "confidence": result["confidence"],
            "prob_ai": result["prob_ai"],
            "threshold": result["threshold"],
            "model_name": str(result.get("model_name", "unknown")),
            "model_version": str(result.get("model_version", "unknown")),
            "inference_mode": result.get("mode", "local"),
            "latency_ms": result.get("latency_ms"),
        }
        with self.engine.begin() as conn:
            return conn.execute(predictions.insert().values(**row)).inserted_primary_key[0]

    def recent(self, limit: int = 100):
        with self.engine.connect() as conn:
            rows = conn.execute(select(predictions).order_by(desc(predictions.c.id)).limit(limit))
            return [dict(r._mapping) for r in rows]

    def stats(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(predictions.c.verdict, func.count()).group_by(predictions.c.verdict))
            return {v: n for v, n in rows}

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
