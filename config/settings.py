import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


COMPLETION_MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
N_SAMPLES = 4
TEMPERATURE = 1.0
TOP_P = 0.9

# Database URLs
QDRANT_URL = "http://40.81.241.185:6333/"
MONGODB_URL = "mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true"

# Database Settings
QDRANT_COLLECTION_NAME = "nContentCol"
MONGODB_DB = "stale"
# MONGODB_COLLECTION = "sebi"

# Research Settings
MAX_ITERATIONS = 1
MAX_CONTEXT_TOKENS = 100000
MAX_EVIDENCE_TOKENS = 500000

# Model Settings
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4.1-2025-04-14"
OPENAI_MINI_MODEL = "gpt-4.1-mini-2025-04-14"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"

# Pricing per 1M tokens (update these with current rates)
MODEL_PRICING = {
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
}

# Omniparse API Configuration
OMNIPARSE_API_URL = "https://14orboabi7zc.share.zrok.io/parse_document"

# # MongoDB Connection Settings
# MONGO_CONFIG = {
#     "maxPoolSize": 10,
#     "minPoolSize": 1,
#     "maxIdleTimeMS": 30000,
#     "connectTimeoutMS": 10000,
#     "serverSelectionTimeoutMS": 10000,
#     "socketTimeoutMS": 20000,
# }