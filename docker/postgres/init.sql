-- PostgreSQL initialization script for AngelaMCP
-- This creates the database and user if they don't exist

-- Create the database if it doesn't exist
SELECT 'CREATE DATABASE angelamcp_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'angelamcp_db')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE angelamcp_db TO angelamcp;

-- Create extensions that might be needed
\c angelamcp_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set timezone
SET timezone = 'UTC';