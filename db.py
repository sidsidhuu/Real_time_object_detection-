import os
import sqlite3
from datetime import datetime


DEFAULT_DB_PATH = os.path.join("data", "detections.db")


def init_db(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def record_detection(session_name, class_name, confidence, detected_at, db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO detections (session_name, class_name, confidence, detected_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_name, class_name, confidence, detected_at),
    )
    conn.commit()
    conn.close()


def list_sessions(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT session_name, MAX(detected_at) AS last_seen, COUNT(*) AS total
        FROM detections
        GROUP BY session_name
        ORDER BY last_seen DESC
        """
    )
    rows = [
        {"session_name": row[0], "last_seen": row[1], "total": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return rows


def list_detections(session_name, limit=100, db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT class_name, confidence, detected_at
        FROM detections
        WHERE session_name = ?
        ORDER BY detected_at DESC
        LIMIT ?
        """,
        (session_name, limit),
    )
    rows = [
        {"class_name": row[0], "confidence": row[1], "timestamp": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return rows


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
