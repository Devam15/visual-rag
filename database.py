import sqlite3
import os
from datetime import datetime

DB_PATH = "./query_logs.db"

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            document_searched TEXT NOT NULL,
            avg_confidence_score REAL,
            num_images_retrieved INTEGER,
            num_chunks_retrieved INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_query(question, answer, document_searched, avg_confidence_score, num_images_retrieved, num_chunks_retrieved):
    """Log a query to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_logs 
        (timestamp, question, answer, document_searched, avg_confidence_score, num_images_retrieved, num_chunks_retrieved)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        answer,
        document_searched,
        avg_confidence_score,
        num_images_retrieved,
        num_chunks_retrieved
    ))
    conn.commit()
    conn.close()

def get_all_logs():
    """Retrieve all query logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_total_queries():
    """Get total number of queries made."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM query_logs")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_avg_confidence():
    """Get average confidence score across all queries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(avg_confidence_score) FROM query_logs")
    avg = cursor.fetchone()[0]
    conn.close()
    return round(avg, 1) if avg else 0

def get_most_queried_documents(limit=5):
    """Get the most frequently queried documents."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document_searched, COUNT(*) as query_count 
        FROM query_logs 
        GROUP BY document_searched 
        ORDER BY query_count DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_recent_queries(limit=10):
    """Get the most recent queries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, question, document_searched, avg_confidence_score 
        FROM query_logs 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_all_logs():
    """Clear all query logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM query_logs")
    conn.commit()
    conn.close()