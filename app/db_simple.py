"""
Simple SQLite database for Florence - Minimal version
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import os


class SimpleFlorenceDB:
    """Minimal SQLite database for Florence persistence"""
    
    def __init__(self, db_path: str = None):
        # Usar path relativo al proyecto
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "florence.db")
        
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database with minimal tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Para acceso por nombre de columna
        
        cursor = self.conn.cursor()
        
        # Tabla 1: Consultas de investigación (lo más importante)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_queries (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla 2: Documentos Knowledge Base (ya tienes JSON, esto es backup)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla 3: Estadísticas de uso (ligero)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                date TEXT PRIMARY KEY,
                query_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para búsqueda rápida
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_queries_date ON research_queries(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kb_date ON kb_documents(created_at)')
        
        self.conn.commit()
        print(f"📦 SQLite database initialized: {self.db_path}")
        print(f"   Size: {self.db_path.stat().st_size / 1024:.1f} KB")
    
    def save_query(self, question: str, response: str = None) -> str:
        """Save a research query - returns query ID"""
        query_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO research_queries (id, question, response) VALUES (?, ?, ?)',
            (query_id, question, response)
        )
        
        # Update daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT OR REPLACE INTO usage_stats (date, query_count, last_updated)
            VALUES (?, COALESCE((SELECT query_count FROM usage_stats WHERE date = ?), 0) + 1, ?)
        ''', (today, today, datetime.now().isoformat()))
        
        self.conn.commit()
        return query_id
    
    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent research queries"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, question, response, created_at FROM research_queries ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        
        rows = cursor.fetchall()
        return [
            {
                'id': row['id'],
                'question': row['question'],
                'response': row['response'],
                'created_at': row['created_at']
            }
            for row in rows
        ]
    
    def save_kb_document(self, content: str, metadata: Dict = None) -> str:
        """Save a Knowledge Base document"""
        doc_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO kb_documents (id, content, metadata) VALUES (?, ?, ?)',
            (doc_id, content, json.dumps(metadata or {}))
        )
        
        self.conn.commit()
        return doc_id
    
    def get_kb_documents(self, limit: int = 50) -> List[Dict]:
        """Get Knowledge Base documents"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, content, metadata, created_at FROM kb_documents ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        
        rows = cursor.fetchall()
        return [
            {
                'id': row['id'],
                'content': row['content'][:200] + '...' if len(row['content']) > 200 else row['content'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'created_at': row['created_at']
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM research_queries')
        query_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM kb_documents')
        doc_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT SUM(query_count) as total FROM usage_stats')
        total_queries = cursor.fetchone()['total'] or 0
        
        return {
            'total_queries': query_count,
            'total_kb_documents': doc_count,
            'total_all_queries': total_queries,
            'database_size_kb': self.db_path.stat().st_size / 1024,
            'database_path': str(self.db_path)
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Cleanup old data to keep database small"""
        cursor = self.conn.cursor()
        
        # Delete queries older than X days
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        cursor.execute('DELETE FROM research_queries WHERE created_at < datetime(?, "unixepoch")', (cutoff_date,))
        deleted_queries = cursor.rowcount
        
        cursor.execute('DELETE FROM kb_documents WHERE created_at < datetime(?, "unixepoch")', (cutoff_date,))
        deleted_docs = cursor.rowcount
        
        # Keep only last 60 days of stats
        cursor.execute('DELETE FROM usage_stats WHERE date < date("now", ?)', (f"-{days_to_keep} days",))
        deleted_stats = cursor.rowcount
        
        self.conn.commit()
        
        # Vacuum to reclaim space
        cursor.execute('VACUUM')
        
        return {
            'deleted_queries': deleted_queries,
            'deleted_documents': deleted_docs,
            'deleted_stats': deleted_stats,
            'new_size_kb': self.db_path.stat().st_size / 1024
        }
    
    def backup(self):
        """Create a simple backup"""
        backup_path = self.db_path.parent / f"florence_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        # Simple copy for SQLite
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        return str(backup_path)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Global instance
florence_db = SimpleFlorenceDB()
