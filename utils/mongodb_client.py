"""
MongoDB Client for Trading Bot - Super TDI + MACD + Super BB Strategy
Version: 3.4.5 - FIXED: Added pandas import for convert_numpy_types
"""

import os
import logging
import time
import pandas as pd  # ← CRITICAL: Added pandas import
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
    from bson import json_util
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    ConnectionFailure = None
    ServerSelectionTimeoutError = None
    OperationFailure = None
    json_util = None

logger = logging.getLogger(__name__)

EMOJI = {
    "START": "🚀",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "WARNING": "⚠️",
    "INFO": "ℹ️",
    "DB": "💾",
    "MONGODB": "🍃",
}


def convert_numpy_types(obj):
    """
    Recursively convert numpy/pandas types to Python native types for MongoDB serialization.
    """
    import numpy as np

    # Handle None
    if obj is None:
        return None

    # Handle dict
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}

    # Handle list or tuple
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Handle numpy types
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.datetime64):
        return obj.astype(datetime)

    # Handle pandas types (NOW WORKS because pd is imported)
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')

    # Handle Python types
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Return as-is for other types
    return obj


class MongoDBClient:
    """
    MongoDB client for storing trading signals.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self.enabled = False
        self._connected = False

        # Load config
        self.uri = self._get_uri()
        self.db_name = os.getenv("MONGODB_DB", os.getenv("MONGO_DB", "trading_bot"))
        self.active_collection = os.getenv("MONGODB_ACTIVE_COLLECTION", "active_signals")
        self.resolved_collection = os.getenv("MONGODB_RESOLVED_COLLECTION", "resolved_signals")
        self.archive_collection = os.getenv("MONGODB_ARCHIVE_COLLECTION", "archive_signals")

        # Connection settings
        self.max_retries = 5
        self.retry_delay = 2
        self.connect_timeout = 5000

        if self.uri:
            self._connect()
        else:
            logger.warning(f"{EMOJI['WARNING']} MONGODB: No URI provided - disabled")

    def _get_uri(self) -> str:
        """Get MongoDB URI from environment or config."""
        # Try various environment variables
        uri = os.getenv("MONGODB_URI", "")
        if uri:
            return uri

        uri = os.getenv("MONGODB_URL", "")
        if uri:
            return uri

        # Build from parts
        host = os.getenv("MONGODB_HOST", os.getenv("MONGO_HOST", "localhost"))
        port = int(os.getenv("MONGODB_PORT", os.getenv("MONGO_PORT", "27017")))
        user = os.getenv("MONGODB_USER", os.getenv("MONGO_USER", ""))
        password = os.getenv("MONGODB_PASSWORD", os.getenv("MONGO_PASS", ""))
        db_name = os.getenv("MONGODB_DB", os.getenv("MONGO_DB", "trading_bot"))

        if user and password:
            return f"mongodb://{user}:{password}@{host}:{port}/{db_name}?authSource=admin"
        elif user:
            return f"mongodb://{user}@{host}:{port}/{db_name}?authSource=admin"
        else:
            return f"mongodb://{host}:{port}/{db_name}"

    def _connect(self):
        """Connect to MongoDB."""
        if not PYMONGO_AVAILABLE:
            logger.warning(f"{EMOJI['WARNING']} MONGODB: PyMongo not installed")
            return

        for attempt in range(self.max_retries):
            try:
                logger.info(f"{EMOJI['MONGODB']} Connecting to MongoDB (attempt {attempt + 1}/{self.max_retries})...")

                self.client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=self.connect_timeout,
                    connectTimeoutMS=self.connect_timeout,
                    socketTimeoutMS=self.connect_timeout,
                )

                # Test connection
                self.client.admin.command('ping')

                self.db = self.client[self.db_name]
                self._connected = True
                self.enabled = True

                # Create indexes
                self._create_indexes()

                logger.info(f"{EMOJI['SUCCESS']} MONGODB: Connected to {self.db_name}")
                return

            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"{EMOJI['WARNING']} MONGODB: Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(f"{EMOJI['ERROR']} MONGODB: Connection error: {e}")
                break

        logger.error(f"{EMOJI['ERROR']} MONGODB: Failed to connect after {self.max_retries} attempts")

    def _create_indexes(self):
        """Create indexes for collections."""
        if not self._connected or self.db is None:
            return

        try:
            active_col = self.db[self.active_collection]
            active_col.create_index("symbol", unique=True)
            active_col.create_index("entry_time")
            active_col.create_index("status")
            active_col.create_index([("symbol", 1), ("status", 1)])

            resolved_col = self.db[self.resolved_collection]
            resolved_col.create_index("symbol")
            resolved_col.create_index("exit_time")
            resolved_col.create_index("status")
            resolved_col.create_index([("symbol", 1), ("exit_time", -1)])

            logger.debug(f"{EMOJI['SUCCESS']} MONGODB: Indexes created")

        except Exception as e:
            logger.warning(f"{EMOJI['WARNING']} MONGODB: Index creation failed: {e}")

    def is_available(self) -> bool:
        """Check if MongoDB is available."""
        return self._connected and self.enabled and self.client is not None

    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Save a signal to MongoDB.

        Args:
            signal_data: Signal data dictionary

        Returns:
            Inserted document ID or None
        """
        if not self.is_available():
            return None

        try:
            # Convert numpy/pandas types to Python types
            cleaned_data = convert_numpy_types(signal_data)

            # Add timestamp
            cleaned_data['saved_at'] = datetime.now()

            collection = self.db[self.active_collection]
            result = collection.insert_one(cleaned_data)

            logger.info(f"{EMOJI['DB']} MONGODB: Signal saved - ID: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_SAVE: Failed - {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_signal_status(self, doc_id: str, status: str, update_data: Dict[str, Any]) -> bool:
        """
        Update signal status.

        Args:
            doc_id: Document ID
            status: New status
            update_data: Additional update data

        Returns:
            True if successful
        """
        if not self.is_available() or not doc_id:
            return False

        try:
            from bson.objectid import ObjectId

            # Convert numpy/pandas types
            cleaned_update = convert_numpy_types(update_data)
            cleaned_update['status'] = status
            cleaned_update['updated_at'] = datetime.now()

            collection = self.db[self.active_collection]

            # Update in active collection
            result = collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": cleaned_update}
            )

            if result.modified_count > 0:
                # Move to resolved collection
                doc = collection.find_one({"_id": ObjectId(doc_id)})
                if doc:
                    resolved_col = self.db[self.resolved_collection]
                    doc['resolved_at'] = datetime.now()
                    resolved_col.insert_one(doc)
                    collection.delete_one({"_id": ObjectId(doc_id)})

                logger.info(f"{EMOJI['DB']} MONGODB: Signal {doc_id} updated to {status}")
                return True

            return False

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_UPDATE: Failed - {e}")
            return False

    def get_active_signals(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get active signals."""
        if not self.is_available():
            return []

        try:
            collection = self.db[self.active_collection]
            query = {"status": "ACTIVE"}
            if symbol:
                query["symbol"] = symbol

            cursor = collection.find(query)
            return list(cursor)

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_GET: Failed - {e}")
            return []

    def get_resolved_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get resolved signals."""
        if not self.is_available():
            return []

        try:
            collection = self.db[self.resolved_collection]
            query = {}
            if symbol:
                query["symbol"] = symbol

            cursor = collection.find(query).sort("exit_time", -1).limit(limit)
            return list(cursor)

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_GET: Failed - {e}")
            return []

    def delete_active_signal(self, symbol: str) -> bool:
        """Delete active signal for a symbol."""
        if not self.is_available():
            return False

        try:
            collection = self.db[self.active_collection]
            result = collection.delete_one({"symbol": symbol})
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_DELETE: Failed - {e}")
            return False

    def clear_all_active(self) -> int:
        """Clear all active signals."""
        if not self.is_available():
            return 0

        try:
            collection = self.db[self.active_collection]
            result = collection.delete_many({})
            logger.info(f"{EMOJI['DB']} MONGODB: Cleared {result.deleted_count} active signals")
            return result.deleted_count

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} MONGODB_CLEAR: Failed - {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'enabled': self.enabled,
            'connected': self._connected,
            'db_name': self.db_name,
        }

        if self.is_available():
            try:
                active_col = self.db[self.active_collection]
                resolved_col = self.db[self.resolved_collection]

                stats['active_count'] = active_col.count_documents({})
                stats['resolved_count'] = resolved_col.count_documents({})
                stats['active_collection'] = self.active_collection
                stats['resolved_collection'] = self.resolved_collection

            except Exception as e:
                stats['error'] = str(e)

        return stats

    def cleanup(self):
        """Clean up MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                self._connected = False
                self.enabled = False
                self.db = None
                logger.info(f"{EMOJI['SUCCESS']} MONGODB: Connection closed")
            except Exception as e:
                logger.warning(f"{EMOJI['WARNING']} MONGODB: Cleanup error: {e}")


# ==================== SINGLETON ====================

mongodb_client = MongoDBClient()

__all__ = [
    "mongodb_client",
    "MongoDBClient",
    "convert_numpy_types",
]