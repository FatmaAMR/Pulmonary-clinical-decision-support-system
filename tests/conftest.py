import os

# إجبار النظام على استخدام SQLite في الذاكرة أثناء أي عملية Testing
os.environ["RAG_STORE_PROVIDER"] = "sqlite_store"
os.environ["RAG_DB_DSN"] = ":memory:"
