import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import json
from typing import Dict, List, Any, Optional
from loguru import logger

class MongoDBConnector:
    """MongoDB connection and data reading class"""
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 username: str = None, password: str = None, 
                 db_name: str = "INS_DB"):
        """
        Initialize MongoDB connector
        
        Args:
            host: MongoDB host address
            port: MongoDB port
            username: Username (optional)
            password: Password (optional)
            db_name: Database name
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db_name = db_name
        self.client = None
        self.db = None
        
    def connect(self) -> bool:
        """
        Connect to MongoDB database
        
        Returns:
            bool: Whether connection was successful
        """
        try:
            # Build connection string
            if self.username and self.password:
                connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}"
            else:
                connection_string = f"mongodb://{self.host}:{self.port}/{self.db_name}"
            
            # Create client connection
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.server_info()
            
            # Get database
            self.db = self.client[self.db_name]
            
            logger.info(f"Successfully connected to MongoDB database: {self.db_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except ServerSelectionTimeoutError as e:
            logger.error(f"Connection timeout: {e}")
            return False
        except Exception as e:
            logger.error(f"Error occurred during connection: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_collections(self) -> List[str]:
        """
        Get all collection names in the database
        
        Returns:
            List[str]: List of collection names
        """
        if self.db is None:
            logger.error("Database not connected")
            return []
        
        try:
            collections = self.db.list_collection_names()
            logger.info(f"Found {len(collections)} collections: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to get collection list: {e}")
            return []
    def create_collection(self, collection_name: str):
        if self.db is None:
            logger.error("Database not connected")
            return False
        try:
            self.db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def read_data(self, collection_name: str, query: Dict = None, 
                  limit: int = None, skip: int = 0) -> List[Dict]:
        """
        Read data from specified collection
        
        Args:
            collection_name: Collection name
            query: Query conditions (optional)
            limit: Limit number of documents returned (optional)
            skip: Number of documents to skip (optional)
            
        Returns:
            List[Dict]: Query result list
        """
        if self.db is None:
            logger.error("Database not connected")
            return []
        
        try:
            collection = self.db[collection_name]
            
            # Build query
            cursor = collection.find(query or {})
            
            # Apply skip and limit
            if skip > 0:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to list
            results = list(cursor)
            
            logger.info(f"Read {len(results)} records from collection '{collection_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            return []
    
    def count_documents(self, collection_name: str, query: Dict = None) -> int:
        """
        Count documents in collection
        
        Args:
            collection_name: Collection name
            query: Query conditions (optional)
            
        Returns:
            int: Document count
        """
        if self.db is None:
            logger.error("Database not connected")
            return 0
        
        try:
            collection = self.db[collection_name]
            count = collection.count_documents(query or {})
            logger.info(f"Collection '{collection_name}' contains {count} records")
            return count
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    def get_sample_data(self, collection_name: str, sample_size: int = 5) -> List[Dict]:
        """
        Get sample data from collection
        
        Args:
            collection_name: Collection name
            sample_size: Sample size
        
        Returns:
            List[Dict]: Sample data list
        """
        return self.read_data(collection_name, limit=sample_size)
    
    def export_to_json(self, collection_name: str, output_file: str, 
                      query: Dict = None, limit: int = None) -> bool:
        """
        Export collection data to JSON file
        
        Args:
            collection_name: Collection name
            output_file: Output file path
            query: Query conditions (optional)
            limit: Limit export count (optional)
            
        Returns:
            bool: Whether export was successful
        """
        try:
            data = self.read_data(collection_name, query, limit)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Data exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False

    def update_db_items(self, collection_name: str, data: Dict):
        self.db[collection_name].update_one({ 'id': data['id'] }, { '$set': data }, upsert=True)

    def batch_insert_db_items(self, collection_name: str, items: List[Dict], batch_size: int = 1000):
        """
        Batch insert data to specified collection. If items length > 1000, insert in batches for efficiency.

        Args:
            collection_name: Collection name
            items: List of data to insert
            batch_size: Number of items per batch (default 1000)
        """
        from pymongo.errors import BulkWriteError

        total_inserted = 0
        total_duplicates = 0

        def do_batch_insert(batch):
            nonlocal total_inserted, total_duplicates
            try:
                result = self.db[collection_name].insert_many(batch, ordered=False)
                inserted_count = len(result.inserted_ids)
                logger.success(f"Inserted {inserted_count} new items, no duplicates")
                total_inserted += inserted_count
            except BulkWriteError as bwe:
                write_errors = bwe.details.get('writeErrors', [])
                duplicate_count = sum(1 for error in write_errors if error.get('code', None) == 11000)
                inserted_count = len(batch) - duplicate_count
                logger.success(f"Inserted {inserted_count} new items, {duplicate_count} duplicates")
                total_inserted += inserted_count
                total_duplicates += duplicate_count
            except Exception as e:
                logger.error(f"Failed to insert many: {e}")

        total = len(items)
        if total > batch_size:
            for i in range(0, total, batch_size):
                do_batch_insert(items[i:i+batch_size])
            logger.info(f"Total inserted: {total_inserted} new items, {total_duplicates} duplicates (batched import)")
        else:
            do_batch_insert(items)
            logger.info(f"Total inserted: {total_inserted} new items, {total_duplicates} duplicates")

    def batch_update_db_items(self, collection_name: str, items: List[Dict], batch_size: int = 1000):
        """
        Batch update data to specified collection (upsert based on _id field). Automatically batches if > batch_size.

        Args:
            collection_name: Collection name
            items: List of data to update (each must contain _id)
            batch_size: Number of items per batch (default 1000)
        """
        from pymongo import UpdateOne
        total_updated = 0
        total_upserted = 0
        total_failed = 0

        def do_batch_update(batch):
            nonlocal total_updated, total_upserted, total_failed
            operations = []
            for item in batch:
                if '_id' in item:
                    operations.append(
                        UpdateOne({'_id': item['_id']}, {'$set': item}, upsert=True)
                    )
                else:
                    logger.warning("Skipping item without _id field.")
            if not operations:
                return
            try:
                result = self.db[collection_name].bulk_write(operations, ordered=False)
                matched = result.matched_count
                upserted = result.upserted_count
                modified = result.modified_count
                logger.success(f"Batch update: matched={matched}, upserted={upserted}, modified={modified}")
                total_updated += modified
                total_upserted += upserted
            except Exception as e:
                logger.error(f"Batch update failed: {e}")
                total_failed += len(batch)

        total = len(items)
        if total == 0:
            logger.info("No data to batch update.")
            return

        if total > batch_size:
            for i in range(0, total, batch_size):
                do_batch_update(items[i:i+batch_size])
            logger.info(f"Batch update stats: Updated/Inserted: {total_updated + total_upserted}, Inserted: {total_upserted}, Failed: {total_failed} (batched)")
        else:
            do_batch_update(items)
            logger.info(f"Batch update stats: Updated/Inserted: {total_updated + total_upserted}, Inserted: {total_upserted}, Failed: {total_failed}")
    def update_one_data(self, collection_name: str, data: Dict, key='_id'):
        if self.db[collection_name] is None:
            self.create_collection(collection_name)
        self.db[collection_name].update_one({ key: data[key] }, { '$set': data }, upsert=True)