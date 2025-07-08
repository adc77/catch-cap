from qdrant_client import QdrantClient as QdrantClientLib, models
from typing import List, Optional
import logging

from config.settings import QDRANT_URL

logger = logging.getLogger(__name__)

class QdrantClient:
    """Client for Qdrant vector database interactions."""
    
    def __init__(self, collection_name: str):
        self.client = QdrantClientLib(url=QDRANT_URL, prefer_grpc=True)
        self.collection_name = collection_name
    
    async def search_points(self, query_embedding: List[float], limit: int = 5):
        """Search for similar points in the collection."""
        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                search_params=models.SearchParams(exact=True)
            )
        except Exception as e:
            logger.error(f"Error searching Qdrant points: {e}")
            raise
    
    async def search_points_with_filter(self, query_embedding: List[float], 
                                 filter_field: str, filter_value: str, 
                                 limit: int = 5):
        """Search for similar points with filter."""
        try:
            filter_ = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_field,
                        match=models.MatchValue(value=filter_value)
                    )
                ]
            )
            
            search_queries = [
                models.QueryRequest(
                    query=query_embedding, 
                    filter=filter_, 
                    limit=limit, 
                    with_payload=True
                )
            ]
            
            return self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=search_queries
            )
        except Exception as e:
            logger.error(f"Error searching Qdrant points with filter: {e}")
            raise