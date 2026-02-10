import os
import sqlite3
import tempfile

from db import init_db, list_detections, list_sessions, record_detection


def _db_path(tmp_dir):
    return os.path.join(tmp_dir, "test.db")


def test_init_db_creates_schema():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = _db_path(tmp_dir)
        init_db(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detections'"
        )
        assert cursor.fetchone() is not None
        conn.close()


def test_record_and_list_detections():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = _db_path(tmp_dir)
        init_db(db_path)

        record_detection("session_a", "person", 91.2, "2026-02-11 10:00:00", db_path)
        record_detection("session_a", "dog", 88.4, "2026-02-11 10:01:00", db_path)

        detections = list_detections("session_a", limit=10, db_path=db_path)
        assert len(detections) == 2
        assert detections[0]["class_name"] in {"person", "dog"}


def test_list_sessions_groups_counts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = _db_path(tmp_dir)
        init_db(db_path)

        record_detection("session_a", "person", 91.2, "2026-02-11 10:00:00", db_path)
        record_detection("session_a", "dog", 88.4, "2026-02-11 10:01:00", db_path)
        record_detection("session_b", "chair", 70.1, "2026-02-11 11:00:00", db_path)

        sessions = list_sessions(db_path=db_path)
        names = {row["session_name"] for row in sessions}
        assert {"session_a", "session_b"}.issubset(names)

        session_a = next(row for row in sessions if row["session_name"] == "session_a")
        assert session_a["total"] == 2
