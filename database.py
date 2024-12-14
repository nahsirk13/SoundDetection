import sqlite3
from datetime import datetime

def init_db():
    """
    Initialize the SQLite database and create the results table if it doesn't exist.
    """
    conn = sqlite3.connect("sound_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sound TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_result(sound):
    """
    Save a sound identification result to the database.
    """
    conn = sqlite3.connect("sound_history.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (sound, timestamp) VALUES (?, ?)", (sound, timestamp))
    conn.commit()
    conn.close()

def query_results(sound=None, month=None):
    """
    Query the database for results based on sound and/or month.
    """
    conn = sqlite3.connect("sound_history.db")
    cursor = conn.cursor()

    query = "SELECT sound, timestamp FROM history WHERE 1=1"
    params = []

    if sound:
        query += " AND sound = ?"
        params.append(sound)
    if month:
        query += " AND strftime('%m', timestamp) = ?"
        params.append(month)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results

