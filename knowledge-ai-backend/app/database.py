import os
import sqlite3

from app.config import DATABASE_PATH, UPLOAD_DIR, document_id_for


def get_connection():
    db_dir = os.path.dirname(DATABASE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        with get_connection() as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }
        if "document_id" not in columns:
            connection = get_connection()
            connection.execute(
                "ALTER TABLE documents ADD COLUMN document_id TEXT"
            )
            connection.commit()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def add_document(filename, size, file_type):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO documents
            (filename, size, file_type, document_id)
            VALUES (?, ?, ?, ?)
            """,
            (filename, size, file_type, document_id_for(filename))
        )
        connection.commit()


def list_documents():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT filename, size, file_type, uploaded_at, document_id
            FROM documents
            ORDER BY uploaded_at DESC
            """
        ).fetchall()
        return [
            {
                **dict(row),
                "document_id": (
                    row["document_id"] or document_id_for(row["filename"])
                ),
            }
            for row in rows
        ]


def delete_document(filename):
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM documents
            WHERE filename = ? OR document_id = ?
            """,
            (filename, filename)
        )
        connection.commit()


def reconcile_documents():
    """Drop database rows that no longer correspond to anything usable.

    ``data/uploads`` and the vector store are ephemeral (lost on redeploy)
    while the SQLite database persists, so rows can outlive the physical
    file. Keeping those rows makes the UI list documents that later fail
    with "Document not found." — this reconciles both and never deletes a
    row whose file or indexed chunks still exist."""
    from app.services.vector_store import vector_store
    indexed_sources = {
        item.get("source") for item in vector_store.all_metadata()
    }
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, filename FROM documents"
        ).fetchall()
        for row in rows:
            missing = not os.path.exists(
                os.path.join(UPLOAD_DIR, row["filename"])
            )
            if missing and row["filename"] not in indexed_sources:
                connection.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (row["id"],)
                )
        connection.commit()


def create_conversation(conversation_id, title):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations
            (conversation_id, title)
            VALUES (?, ?)
            """,
            (conversation_id, title)
        )
        connection.commit()


def update_conversation_title(conversation_id, title):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (title, conversation_id)
        )
        connection.commit()


def add_message(conversation_id, role, content):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content)
        )
        connection.commit()


def get_messages(conversation_id, limit=10):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit)
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


def get_conversations():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def delete_conversation(conversation_id):
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        )
        connection.execute(
            """
            DELETE FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        )
        connection.commit()

