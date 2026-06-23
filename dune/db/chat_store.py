import sqlite3
import os
from typing import List, Dict, Any
import datetime
import uuid

class ChatStore:
    def __init__(self, db_path: str = "dune_chat.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_session(self, title: str = "New Chat") -> Dict[str, Any]:
        """Create a new chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        
        cursor.execute(
            'INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
            (session_id, title, now, now)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now
        }

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions ordered by recently updated."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions ORDER BY updated_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get a single session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Foreign key cascading isn't enabled by default in sqlite3 python module unless PRAGMA foreign_keys = ON
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0
        
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update a session's title."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.datetime.utcnow().isoformat()
        cursor.execute('UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?', (title, now, session_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0

    def add_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """Add a message to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        msg_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        
        cursor.execute(
            'INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)',
            (msg_id, session_id, role, content, now)
        )
        
        # Update session timestamp
        cursor.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (now, session_id))
        
        conn.commit()
        conn.close()
        
        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now
        }

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific session ordered by creation time."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
