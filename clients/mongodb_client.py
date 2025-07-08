from pymongo import MongoClient
from typing import Optional, Dict, Any
import logging

from config.settings import MONGODB_URL, MONGODB_DB

logger = logging.getLogger(__name__)

class MongoDBClient:
    """Client for MongoDB interactions."""
    
    def __init__(self, db_name: str = MONGODB_DB):
        self.client = MongoClient(MONGODB_URL)
        self.db = self.client[db_name]
    
    def find_document_by_url(self, collection_name: str, url: str) -> Optional[Dict[str, Any]]:
        """Find document by PDF URL."""
        try:
            collection = self.db[collection_name]
            return collection.find_one(
                {"pdfUrls": url},
                max_time_ms=10000  # 10 second timeout
            )
        except Exception as e:
            logger.error(f"Error finding document in MongoDB: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()