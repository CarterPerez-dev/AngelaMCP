# MACP Setup Guide

Complete step-by-step installation guide for Debian/Ubuntu Linux.

## 📋 System Requirements

- **OS**: Debian 11+ or Ubuntu 20.04+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Claude Code**: Already installed and configured

## 🔧 Step 1: System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib postgresql-client

# Install Redis
sudo apt install -y redis-server

# Install additional dependencies
sudo apt install -y git curl wget build-essential libpq-dev

# Verify installations
python3.10 --version
psql --version
redis-cli --version
claude --version  # Should show Claude Code version
```

## 🗄️ Step 2: Database Setup

### PostgreSQL Configuration

```bash
# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE USER macp_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE macp_db OWNER macp_user;
GRANT ALL PRIVILEGES ON DATABASE macp_db TO macp_user;
\q
EOF

# Test connection
PGPASSWORD='your_secure_password' psql -h localhost -U macp_user -d macp_db -c '\l'
```

### Redis Configuration

```bash
# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Configure Redis for persistence (optional but recommended)
sudo bash -c 'echo "appendonly yes" >> /etc/redis/redis.conf'
sudo systemctl restart redis-server

# Test Redis
redis-cli ping  # Should return PONG
```

## 🐍 Step 3: Python Environment

```bash
# Create project directory
mkdir -p ~/projects/macp
cd ~/projects/macp

# Clone the repository (or create structure)
git clone https://github.com/yourusername/macp.git .

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## 📦 Step 4: Install Dependencies

Create `requirements.txt`:

```bash
cat > requirements.txt << 'EOF'
# Core Dependencies
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# API Clients
openai==1.12.0
google-generativeai==0.3.2
anthropic==0.18.1

# Database
sqlalchemy==2.0.25
asyncpg==0.29.0
psycopg2-binary==2.9.9
redis==5.0.1
alembic==1.13.1

# Async Support
asyncio==3.4.3
aiohttp==3.9.3
aiofiles==23.2.1

# Terminal UI
rich==13.7.0
prompt-toolkit==3.0.43
click==8.1.7

# Utilities
pyyaml==6.0.1
structlog==24.1.0
tenacity==8.2.3

# Development
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
ruff==0.1.14
mypy==1.8.0
black==23.12.1

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.3
EOF

# Install all dependencies
pip install -r requirements.txt
```

## ⚙️ Step 5: Configuration

### Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Or create new one
cat > .env << 'EOF'
# Application Settings
APP_NAME=MACP
APP_ENV=production
LOG_LEVEL=INFO

# Claude Code Configuration
CLAUDE_CODE_PATH=/usr/local/bin/claude
CLAUDE_CODE_TIMEOUT=300

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=o3-mini
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7

# Google Gemini Configuration
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_MODEL=gemini-2.5-pro-preview-06-05
GEMINI_MAX_TOKENS=4096
GEMINI_TEMPERATURE=0.7

# Database Configuration
DATABASE_URL=postgresql://macp_user:your_secure_password@localhost:5432/macp_db
REDIS_URL=redis://localhost:6379/0

# API Rate Limiting
RATE_LIMIT_OPENAI=60  # requests per minute
RATE_LIMIT_GEMINI=60  # requests per minute

# Session Configuration
SESSION_TIMEOUT=3600  # 1 hour
MAX_CONVERSATION_LENGTH=100  # messages

# Feature Flags
ENABLE_COST_TRACKING=true
ENABLE_PARALLEL_EXECUTION=true
ENABLE_DEBATE_MODE=true
EOF

# Secure the environment file
chmod 600 .env
```

### Verify API Keys

```bash
# Create verification script
cat > scripts/verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""Verify MACP setup and API connectivity."""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_claude_code():
    """Check Claude Code installation."""
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Claude Code is installed")
            return True
    except FileNotFoundError:
        pass
    print("❌ Claude Code not found. Please install it first.")
    return False

def check_env_vars():
    """Check required environment variables."""
    required = [
        'OPENAI_API_KEY',
        'GOOGLE_API_KEY',
        'DATABASE_URL',
        'REDIS_URL'
    ]
    
    missing = []
    for var in required:
        if not os.getenv(var):
            missing.append(var)
            print(f"❌ Missing: {var}")
        else:
            print(f"✅ Found: {var}")
    
    return len(missing) == 0

def check_database():
    """Check database connectivity."""
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        db_url = os.getenv('DATABASE_URL')
        url = urlparse(db_url)
        
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            user=url.username,
            password=url.password,
            database=url.path[1:]
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

def check_redis():
    """Check Redis connectivity."""
    try:
        import redis
        r = redis.from_url(os.getenv('REDIS_URL'))
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("MACP Setup Verification\n" + "="*50)
    
    checks = [
        check_claude_code(),
        check_env_vars(),
        check_database(),
        check_redis()
    ]
    
    if all(checks):
        print("\n✅ All checks passed! MACP is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
EOF

chmod +x scripts/verify_setup.py

# Run verification
python scripts/verify_setup.py
```

## 🗃️ Step 6: Database Initialization

```bash
# Create database schema
cat > scripts/init_db.py << 'EOF'
#!/usr/bin/env python3
"""Initialize MACP database schema."""

import asyncio
from sqlalchemy import create_engine
from src.persistence.models import Base
from config.settings import settings

def init_database():
    """Create all database tables."""
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("✅ Database schema created successfully")

if __name__ == "__main__":
    init_database()
EOF

# Run database initialization
python scripts/init_db.py
```

## 🚀 Step 7: First Run

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run MACP
python -m src.main

# Or use the Makefile
make run
```

## 🧪 Step 8: Verify Installation

```bash
# Run tests
make test

# Test individual agents
python scripts/test_agents.py

# Check logs
tail -f logs/macp.log
```

## 🐛 Troubleshooting

### Common Issues

1. **Claude Code not found**
   ```bash
   # Verify Claude is in PATH
   which claude
   
   # Add to PATH if needed
   echo 'export PATH="$PATH:/path/to/claude"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **PostgreSQL connection refused**
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql
   
   # Check pg_hba.conf allows local connections
   sudo nano /etc/postgresql/14/main/pg_hba.conf
   # Ensure you have: local all all md5
   ```

3. **Redis connection refused**
   ```bash
   # Check Redis is running
   sudo systemctl status redis-server
   
   # Test with redis-cli
   redis-cli ping
   ```

4. **API key errors**
   ```bash
   # Verify keys are set
   python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10] + '...')"
   ```

## 📚 Next Steps

1. Read [ARCHITECTURE.md](./docs/ARCHITECTURE.md) to understand the system design
2. Check [docs/examples/](./docs/examples/) for usage examples
3. Configure your preferred settings in `.env`
4. Start building with MACP!

---

Need help? Check our [FAQ](./docs/FAQ.md) or open an issue on GitHub.
