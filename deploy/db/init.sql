CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng test kiểm tra pgvector và ghi dữ liệu
CREATE TABLE IF NOT EXISTS test_embeddings (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

