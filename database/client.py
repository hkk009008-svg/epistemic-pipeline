import os
import uuid
import logging
from typing import Optional, List
import psycopg2
import psycopg2.extras
import config

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection():
    # If DATABASE_URL is not set, psycopg2 will raise an error unless handled
    if not config.DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(config.DATABASE_URL)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to DB: {e}")
        return None


def init_db():
    if not os.path.exists(SCHEMA_PATH):
        logging.error(f"Schema file not found at {SCHEMA_PATH}")
        return

    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    conn = get_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
    finally:
        conn.close()


def insert_entity(name: str, entity_type: str = "concept") -> Optional[str]:
    """Inserts or retrieves an entity and returns its ID."""
    conn = get_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id FROM entities WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row["id"]

        entity_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO entities (id, name, type) VALUES (%s, %s, %s)",
            (entity_id, name, entity_type),
        )
        conn.commit()
        return entity_id
    except Exception as e:
        logging.error(f"Error inserting entity: {e}")
        return None
    finally:
        conn.close()


def insert_source(url: Optional[str], domain: Optional[str] = None) -> Optional[str]:
    if not url:
        return None
    conn = get_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id FROM sources WHERE url = %s", (url,))
        row = cur.fetchone()
        if row:
            return row["id"]

        source_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO sources (id, url, domain) VALUES (%s, %s, %s)",
            (source_id, url, domain),
        )
        conn.commit()
        return source_id
    except Exception as e:
        logging.error(f"Error inserting source: {e}")
        return None
    finally:
        conn.close()


def insert_claim(
    subject_name: str,
    relation: str,
    object_name: str,
    source_url: Optional[str] = None,
    confidence: str = "High",
    original_text: Optional[str] = None,
) -> Optional[str]:
    """Inserts a claim into the knowledge graph."""
    subject_id = insert_entity(subject_name)
    object_id = insert_entity(object_name)
    source_id = insert_source(source_url) if source_url else None

    if not subject_id or not object_id:
        return None

    conn = get_connection()
    if not conn:
        return None
    try:
        claim_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO claims 
               (id, subject_id, relation, object_id, source_id, confidence, original_text) 
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                claim_id,
                subject_id,
                relation,
                object_id,
                source_id,
                confidence,
                original_text,
            ),
        )
        conn.commit()
        return claim_id
    except Exception as e:
        logging.error(f"Error inserting claim: {e}")
        return None
    finally:
        conn.close()


def find_knowledge(query: str) -> List[dict]:
    """Basic retrieval of claims related to terms in the query."""
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        words = [w.strip("?,.").lower() for w in query.split() if len(w) > 3]
        if not words:
            return []

        like_words = [f"%{word}%" for word in words]
        like_clauses = " OR ".join(["lower(name) LIKE %s"] * len(words))

        cur.execute(f"SELECT id FROM entities WHERE {like_clauses}", like_words)
        entity_ids = [row["id"] for row in cur.fetchall()]

        if not entity_ids:
            return []

        placeholders = ",".join("%s" for _ in entity_ids)
        query_sql = f"""
            SELECT c.id, s.name as subject, c.relation, o.name as object, c.confidence, c.original_text, src.url
            FROM claims c
            JOIN entities s ON c.subject_id = s.id
            JOIN entities o ON c.object_id = o.id
            LEFT JOIN sources src ON c.source_id = src.id
            WHERE c.subject_id IN ({placeholders}) OR c.object_id IN ({placeholders})
        """
        cur.execute(query_sql, tuple(entity_ids + entity_ids))

        results = []
        for row in cur.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "relation": row["relation"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "original_text": row["original_text"],
                    "source_url": row["url"],
                }
            )
        return results
    except Exception as e:
        logging.error(f"Error finding knowledge: {e}")
    finally:
        conn.close()
    return []


def detect_collision(subject: str, relation: str, object_name: str) -> bool:
    """
    Checks if a claim logically contradicts the graph.
    In V1, we simply flag if the *exact opposite* relationship exists.
    """
    conn = get_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT id FROM entities WHERE name = %s", (subject,))
        s_row = cur.fetchone()
        if not s_row:
            return False

        cur.execute("SELECT id FROM entities WHERE name = %s", (object_name,))
        o_row = cur.fetchone()
        if not o_row:
            return False

        cur.execute(
            """
            SELECT relation, original_text FROM claims 
            WHERE subject_id = %s AND object_id = %s
        """,
            (s_row["id"], o_row["id"]),
        )

        existing_claims = cur.fetchall()
        for row in existing_claims:
            existing_rel = row["relation"].lower()
            new_rel = relation.lower()

            if ("not" in existing_rel and "not" not in new_rel) or (
                "not" in new_rel and "not" not in existing_rel
            ):
                return True

        return False
    except Exception as e:
        logging.error(f"Error detecting collision: {e}")
    finally:
        conn.close()
    return False


def get_full_graph() -> dict:
    """Returns the full graph (nodes and edges) for UI visualization."""
    conn = get_connection()
    if not conn:
        return {"nodes": [], "edges": []}

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, name, type FROM entities")
        nodes = [
            {"id": r["id"], "name": r["name"], "type": r["type"]}
            for r in cur.fetchall()
        ]

        cur.execute(
            "SELECT id, subject_id, relation, object_id, confidence FROM claims"
        )
        edges = [
            {
                "id": r["id"],
                "source": r["subject_id"],
                "target": r["object_id"],
                "label": r["relation"],
                "confidence": r["confidence"],
            }
            for r in cur.fetchall()
        ]

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logging.error(f"Error getting full graph: {e}")
    finally:
        conn.close()
    return {"nodes": [], "edges": []}


if __name__ == "__main__":
    init_db()
