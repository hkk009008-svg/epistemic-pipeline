CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE,
    domain TEXT,
    trust_score REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    object_id TEXT NOT NULL,
    source_id TEXT,
    confidence TEXT,
    original_text TEXT,
    ttl_expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(subject_id) REFERENCES entities(id),
    FOREIGN KEY(object_id) REFERENCES entities(id),
    FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    rejected_claim_text TEXT NOT NULL,
    conflicting_claim_id TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(conflicting_claim_id) REFERENCES claims(id)
);
