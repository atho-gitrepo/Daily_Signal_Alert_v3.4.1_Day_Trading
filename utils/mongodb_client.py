"""
MongoDB Client for Trading Bot - Super TDI + MACD + Super BB Strategy
Version: 3.4.5 - FIXED: Duplicate prevention, proper signal resolution, infinite loop fix
"""

import os
import logging
import time
import json
import traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure, DuplicateKeyError
    from bson import json_util
    from bson.objectid import ObjectId
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    ConnectionFailure = None
    ServerSelectionTimeoutError = None
    OperationFailure = None
    DuplicateKeyError = None
    json_util = None
    ObjectId = None

logger = logging.getLogger(__name__)

EMOJI = {
    "START": "🚀",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "WARNING": "⚠️",
    "INFO": "ℹ️",
    "DB": "💾",
    "MONGODB": "🍃",
    "SAVE": "💾",
    "DELETE": "🗑️",
    "UPDATE": "🔄",
    "FIND": "🔍",
    "RESOLVED": "✅",
    "RESTORE": "♻️",
    "ACTIVE": "🔴",
    "FAILOVER": "💾",
    "CONVERT": "🔄",
    "FIXED": "🔧",
    "VERIFY": "✅",
}


def convert_numpy_types(obj):
    """
    Recursively convert numpy/pandas types to Python native types for MongoDB serialization.
    COMPREHENSIVE FIX: Handles all common types.
    """
    # Handle None
    if obj is None:
        return None

    # Handle numpy types
    if NUMPY_AVAILABLE and np is not None:
        try:
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.datetime64):
                return obj.astype(datetime)
            elif isinstance(obj, np.generic):
                return obj.item()
        except Exception as e:
            logger.debug(f"NumPy conversion warning: {e}")

    # Handle pandas types
    if PANDAS_AVAILABLE and pd is not None:
        try:
            import pandas as pd
            if isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Timestamp):
                return obj.to_pydatetime()
        except Exception as e:
            logger.debug(f"Pandas conversion warning: {e}")

    # Handle datetime
    if isinstance(obj, datetime):
        return obj

    # Handle dict - RECURSIVE
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}

    # Handle list or tuple - RECURSIVE
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Handle primitive types
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except:
            return str(obj)

    # Handle sets
    if isinstance(obj, set):
        return list(obj)

    # Fallback
    try:
        return str(obj)
    except:
        return f"<Unserializable: {type(obj).__name__}>"


class MongoDBClient:
    """
    MongoDB client for storing trading signals.
    Version 3.4.5: Fixed duplicate prevention and signal resolution.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self.enabled = False
        self._connected = False
        self.version = "3.4.5"
        self._last_error = None

        # Collection names
        self.db_name = os.getenv("MONGODB_DB", os.getenv("MONGO_DB", "trading_bot_dca"))
        self.active_collection = os.getenv("MONGODB_ACTIVE_COLLECTION", "active_signals")
        self.resolved_collection = os.getenv("MONGODB_RESOLVED_COLLECTION", "resolved_signals")
        self.archive_collection = os.getenv("MONGODB_ARCHIVE_COLLECTION", "archive_signals")

        # Connection settings
        self.max_retries = 5
        self.retry_delay = 2
        self.connect_timeout = 5000

        # Metrics
        self.metrics = {
            "signals_saved": 0,
            "signals_resolved": 0,
            "active_deletions": 0,
            "errors": 0,
            "duplicates_prevented": 0,
        }

        # Get URI and connect
        self.uri = self._get_uri()

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

        uri = os.getenv("MONGO_URI", "")
        if uri:
            return uri

        # Build from parts
        host = os.getenv("MONGODB_HOST", os.getenv("MONGO_HOST", "localhost"))
        port = int(os.getenv("MONGODB_PORT", os.getenv("MONGO_PORT", "27017")))
        user = os.getenv("MONGODB_USER", os.getenv("MONGO_USER", ""))
        password = os.getenv("MONGODB_PASSWORD", os.getenv("MONGO_PASS", ""))
        db_name = os.getenv("MONGODB_DB", os.getenv("MONGO_DB", "trading_bot_dca"))

        if user and password:
            if 'mongodb.net' in host:
                return f"mongodb+srv://{user}:{password}@{host}/{db_name}?retryWrites=true&w=majority"
            else:
                return f"mongodb://{user}:{password}@{host}:{port}/{db_name}?authSource=admin"
        elif user:
            return f"mongodb://{user}@{host}:{port}/{db_name}?authSource=admin"
        else:
            return f"mongodb://{host}:{port}/{db_name}"

    def _connect(self):
        """Connect to MongoDB with retry logic."""
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

                # Create indexes (with error handling for duplicate index names)
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
        """Create indexes for collections with error handling."""
        if not self._connected or self.db is None:
            return

        try:
            # Active signals indexes
            active_col = self.db[self.active_collection]
            
            # Check if unique index exists before trying to create
            existing_indexes = active_col.index_information()
            
            # Only create symbol index if it doesn't exist or isn't unique
            if 'symbol_1' not in existing_indexes:
                active_col.create_index([("symbol", ASCENDING)])
            else:
                logger.debug("Index symbol_1 already exists, skipping creation")
            
            # Other indexes
            if 'entry_time_-1' not in existing_indexes:
                active_col.create_index([("entry_time", DESCENDING)])
            
            if 'status_1' not in existing_indexes:
                active_col.create_index([("status", ASCENDING)])
            
            if 'symbol_1_status_1' not in existing_indexes:
                active_col.create_index([("symbol", ASCENDING), ("status", ASCENDING)])

            # Resolved signals indexes
            resolved_col = self.db[self.resolved_collection]
            existing_resolved = resolved_col.index_information()
            
            if 'symbol_1' not in existing_resolved:
                resolved_col.create_index([("symbol", ASCENDING)])
            
            if 'exit_time_-1' not in existing_resolved:
                resolved_col.create_index([("exit_time", DESCENDING)])

            logger.info(f"{EMOJI['SUCCESS']} MONGODB: Indexes created/verified")

        except OperationFailure as e:
            if 'IndexKeySpecsConflict' in str(e) or 'already exists' in str(e):
                logger.debug(f"{EMOJI['INFO']} Index already exists (non-critical): {e}")
            else:
                logger.warning(f"{EMOJI['WARNING']} MONGODB: Index creation failed: {e}")
        except Exception as e:
            logger.warning(f"{EMOJI['WARNING']} MONGODB: Index creation failed: {e}")

    def is_available(self) -> bool:
        """Check if MongoDB is available."""
        if not self._connected or not self.enabled or self.client is None or self.db is None:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except:
            self._connected = False
            return False

    # ==================== FIXED: SIGNAL OPERATIONS ====================

    def get_active_signal_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get active signal by symbol - with proper type handling.
        CRITICAL: Used for duplicate prevention.
        """
        if not self.is_available():
            return None

        try:
            active_col = self.db[self.active_collection]
            signal = active_col.find_one({
                'symbol': symbol,
                'status': {'$in': ['ACTIVE', 'AI_WAITING']}
            })
            
            if signal:
                if '_id' in signal:
                    signal['_id'] = str(signal['_id'])
                logger.debug(f"Found active signal for {symbol}: {signal.get('_id')}")
                return signal
            return None

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} get_active_signal_by_symbol failed: {e}")
            return None

    def save_signal_upsert(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Save signal using upsert to prevent duplicate key errors.
        PRIMARY METHOD for saving signals.
        """
        if not self.is_available():
            logger.warning(f"{EMOJI['WARNING']} MongoDB not available - cannot save")
            return None

        try:
            # Convert types
            signal_data = convert_numpy_types(signal_data)
            
            # Add metadata
            signal_data['created_at'] = datetime.now().isoformat()
            signal_data['updated_at'] = datetime.now().isoformat()
            signal_data['strategy_version'] = self.version
            
            # Generate ID if not present
            if '_id' not in signal_data or not signal_data['_id']:
                signal_data['_id'] = self._generate_id(signal_data)
            
            doc_id = signal_data['_id']
            symbol = signal_data.get('symbol')
            
            if not symbol:
                logger.warning("No symbol in signal data")
                return None
            
            active_col = self.db[self.active_collection]
            
            logger.info(f"{EMOJI['SAVE']} Saving signal for {symbol} using upsert...")
            logger.debug(f"  doc_id: {doc_id}")
            logger.debug(f"  conditions: {signal_data.get('conditions_met', 0)}/{signal_data.get('conditions_total', 5)}")
            
            # ===== FIXED: USE UPSERT TO PREVENT DUPLICATE KEY ERRORS =====
            result = active_col.update_one(
                {'symbol': symbol, 'status': {'$in': ['ACTIVE', 'AI_WAITING']}},
                {'$set': signal_data},
                upsert=True
            )

            if result.acknowledged:
                self.metrics["signals_saved"] += 1
                
                if result.upserted_id:
                    logger.info(f"{EMOJI['SUCCESS']} ✅ New signal saved: {doc_id} for {symbol}")
                else:
                    logger.info(f"{EMOJI['UPDATE']} ✅ Existing signal updated: {doc_id} for {symbol}")
                    self.metrics["duplicates_prevented"] += 1
                
                return doc_id
            else:
                logger.warning(f"Save not acknowledged for {symbol}")
                return None

        except DuplicateKeyError as e:
            self.metrics["duplicates_prevented"] += 1
            logger.warning(f"{EMOJI['WARNING']} Duplicate key prevented: {e}")
            # Try to recover by getting existing signal
            try:
                symbol = signal_data.get('symbol')
                if symbol:
                    existing = self.get_active_signal_by_symbol(symbol)
                    if existing:
                        logger.info(f"{EMOJI['SUCCESS']} Found existing signal: {existing.get('_id')}")
                        return str(existing['_id'])
            except:
                pass
            return None
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"{EMOJI['ERROR']} save_signal_upsert failed: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None

    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Legacy save method - now calls save_signal_upsert.
        Kept for backward compatibility.
        """
        return self.save_signal_upsert(signal_data)

    def move_to_resolved(self, doc_id: str, signal_data: Dict[str, Any]) -> bool:
        """
        Move a signal to resolved collection and delete from active.
        COMPREHENSIVE FIX: Properly handles duplicates and prevents infinite loops.
        """
        # Convert types
        signal_data = convert_numpy_types(signal_data)
        
        logger.info(f"{EMOJI['RESOLVED']} Moving {doc_id} to resolved...")

        if not self.is_available():
            logger.warning(f"{EMOJI['WARNING']} MongoDB not available - cannot move to resolved")
            return False

        try:
            # ===== FIXED: Check if already in resolved =====
            resolved_col = self.db[self.resolved_collection]
            existing_resolved = resolved_col.find_one({'_id': doc_id})
            
            if existing_resolved:
                logger.info(f"{EMOJI['INFO']} Signal {doc_id} already in resolved - skipping duplicate move")
                # Still try to delete from active
                active_col = self.db[self.active_collection]
                delete_result = active_col.delete_one({'_id': doc_id})
                if delete_result.deleted_count > 0:
                    self.metrics["active_deletions"] += 1
                    logger.info(f"{EMOJI['DELETE']} ✅ Deleted from active: {doc_id}")
                return True

            # ===== FIXED: Check if signal exists in active =====
            active_col = self.db[self.active_collection]
            active_signal = active_col.find_one({'_id': doc_id})
            
            if not active_signal:
                logger.warning(f"{EMOJI['WARNING']} Signal {doc_id} not in active collection")
                # Check if we can find by symbol
                symbol = signal_data.get('symbol')
                if symbol:
                    active_signal = active_col.find_one({'symbol': symbol, 'status': {'$ne': 'RESOLVED'}})
                    if active_signal:
                        logger.info(f"{EMOJI['INFO']} Found {symbol} in active collection by symbol")
                        doc_id = str(active_signal.get('_id'))
                    else:
                        logger.info(f"{EMOJI['INFO']} Signal {symbol} not in active - may already be resolved")
                        return True
                else:
                    return True

            # Prepare data for resolved
            signal_data['_id'] = doc_id
            signal_data['resolved_at'] = datetime.now().isoformat()
            signal_data['strategy_version'] = self.version
            signal_data['original_doc_id'] = doc_id

            # ===== STEP 1: SAVE TO RESOLVED COLLECTION =====
            logger.info(f"{EMOJI['DB']} Saving to resolved: {doc_id}")
            logger.debug(f"  Conditions: {signal_data.get('conditions_met', 'N/A')}/{signal_data.get('conditions_total', 5)}")
            logger.debug(f"  PnL: ${signal_data.get('pnl', 0):.2f}")
            
            resolved_result = resolved_col.update_one(
                {'_id': doc_id},
                {'$set': signal_data},
                upsert=True
            )

            if resolved_result.acknowledged:
                self.metrics["signals_resolved"] += 1
                logger.info(f"{EMOJI['RESOLVED']} ✅ Saved to resolved: {doc_id}")
            else:
                logger.warning(f"{EMOJI['WARNING']} Failed to save {doc_id} to resolved")
                return False

            # ===== STEP 2: DELETE FROM ACTIVE =====
            try:
                delete_result = active_col.delete_one({'_id': doc_id})
                
                if delete_result.deleted_count > 0:
                    self.metrics["active_deletions"] += 1
                    logger.info(f"{EMOJI['DELETE']} ✅ Deleted from active: {doc_id}")
                else:
                    # Try to delete by symbol as fallback
                    symbol = signal_data.get('symbol')
                    if symbol:
                        alt_delete = active_col.delete_one({'symbol': symbol, 'status': {'$ne': 'RESOLVED'}})
                        if alt_delete.deleted_count > 0:
                            self.metrics["active_deletions"] += alt_delete.deleted_count
                            logger.info(f"{EMOJI['DELETE']} ✅ Deleted {symbol} from active (fallback)")
                        else:
                            logger.warning(f"{EMOJI['WARNING']} Failed to delete {doc_id} from active")
            except Exception as e:
                logger.warning(f"{EMOJI['WARNING']} Delete from active failed: {e}")

            # ===== STEP 3: UPDATE ARCHIVE =====
            try:
                archive_col = self.db[self.archive_collection]
                archive_col.update_one(
                    {'_id': doc_id},
                    {'$set': signal_data},
                    upsert=True
                )
                logger.debug(f"{EMOJI['DB']} Archived: {doc_id}")
            except Exception as e:
                logger.debug(f"Archive save failed: {e}")

            return True

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"{EMOJI['ERROR']} move_to_resolved failed: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return False

    def update_signal_status(self, doc_id: str, status: str, update_data: Dict[str, Any]) -> bool:
        """
        Update signal status - calls move_to_resolved for terminal statuses.
        """
        if not doc_id:
            return False

        # Convert types
        update_data = convert_numpy_types(update_data)
        update_data['status'] = status
        
        terminal_statuses = ['PROFIT', 'LOSS', 'BREAK_EVEN', 'CLOSED', 'EXIT_1H', 'EXIT_TARGET']
        
        if status in terminal_statuses:
            return self.move_to_resolved(doc_id, update_data)
        else:
            # Non-terminal: update in active
            if not self.is_available():
                logger.warning(f"{EMOJI['WARNING']} MongoDB not available")
                return False
            
            try:
                active_col = self.db[self.active_collection]
                result = active_col.update_one(
                    {'_id': doc_id},
                    {'$set': update_data}
                )
                return result.acknowledged
            except Exception as e:
                logger.error(f"{EMOJI['ERROR']} update_signal_status failed: {e}")
                return False

    # ==================== GET OPERATIONS ====================

    def get_active_signals(self, limit: int = 100) -> Dict[str, Dict]:
        """Get all active signals from active collection."""
        if not self.is_available():
            return {}

        try:
            active_col = self.db[self.active_collection]
            signals = active_col.find(
                {'status': {'$in': ['ACTIVE', 'AI_WAITING']}},
                limit=limit
            ).sort('entry_time', DESCENDING)

            result = {}
            for signal in signals:
                doc_id = str(signal.get('_id', ''))
                if doc_id:
                    signal['_id'] = doc_id
                    result[doc_id] = signal

            logger.debug(f"{EMOJI['ACTIVE']} Found {len(result)} active signals")
            return result

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} get_active_signals failed: {e}")
            return {}

    def get_resolved_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get resolved signals."""
        if not self.is_available():
            return []

        try:
            resolved_col = self.db[self.resolved_collection]
            query = {}
            if symbol:
                query["symbol"] = symbol

            cursor = resolved_col.find(query).sort("exit_time", -1).limit(limit)
            result = []
            for signal in cursor:
                if '_id' in signal:
                    signal['_id'] = str(signal['_id'])
                result.append(signal)
            return result

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} get_resolved_signals failed: {e}")
            return []

    def cleanup_orphaned_active_signals(self) -> int:
        """
        Clean up orphaned active signals that should have been deleted.
        COMPREHENSIVE FIX: Properly handles all orphan cases.
        """
        if not self.is_available():
            return 0

        try:
            active_col = self.db[self.active_collection]
            resolved_col = self.db[self.resolved_collection]
            
            # Find signals that should be resolved
            terminal_statuses = ['PROFIT', 'LOSS', 'BREAK_EVEN', 'CLOSED', 'RESOLVED', 'EXIT_1H', 'EXIT_TARGET']
            
            # Also find signals with exit_time set
            orphaned = active_col.find({
                '$or': [
                    {'status': {'$in': terminal_statuses}},
                    {'exit_time': {'$exists': True}},
                    {'resolved_at': {'$exists': True}},
                ]
            })
            
            orphaned_list = list(orphaned)
            deleted_count = 0

            if not orphaned_list:
                return 0

            logger.info(f"{EMOJI['ACTIVE']} Found {len(orphaned_list)} orphaned signals to clean up")

            for signal in orphaned_list:
                doc_id = signal.get('_id')
                symbol = signal.get('symbol', 'UNKNOWN')
                
                if not doc_id:
                    continue
                    
                try:
                    # Check if already in resolved
                    existing_resolved = resolved_col.find_one({'_id': doc_id})
                    
                    if existing_resolved:
                        # Already in resolved, just delete from active
                        delete_result = active_col.delete_one({'_id': doc_id})
                        if delete_result.deleted_count > 0:
                            deleted_count += 1
                            self.metrics["active_deletions"] += 1
                            logger.info(f"{EMOJI['DELETE']} Cleaned up {symbol}: {doc_id} (already in resolved)")
                        continue
                    
                    # Move to resolved first
                    signal_data = signal.copy()
                    if '_id' in signal_data:
                        del signal_data['_id']
                    signal_data['status'] = 'RESOLVED'
                    signal_data['resolved_at'] = datetime.now().isoformat()
                    signal_data['resolved_by'] = 'cleanup'
                    signal_data['strategy_version'] = self.version
                    
                    resolved_result = resolved_col.update_one(
                        {'_id': doc_id},
                        {'$set': signal_data},
                        upsert=True
                    )
                    
                    if resolved_result.acknowledged:
                        # Now delete from active
                        delete_result = active_col.delete_one({'_id': doc_id})
                        if delete_result.deleted_count > 0:
                            deleted_count += 1
                            self.metrics["active_deletions"] += 1
                            self.metrics["signals_resolved"] += 1
                            logger.info(f"{EMOJI['DELETE']} Cleaned up {symbol}: {doc_id}")
                        else:
                            # Try to delete by symbol as fallback
                            alt_delete = active_col.delete_one({'symbol': symbol, '_id': doc_id})
                            if alt_delete.deleted_count > 0:
                                deleted_count += 1
                                self.metrics["active_deletions"] += 1
                                logger.info(f"{EMOJI['DELETE']} Cleaned up {symbol}: {doc_id} (by symbol)")
                    else:
                        logger.warning(f"{EMOJI['WARNING']} Failed to move {symbol}: {doc_id} to resolved")
                        
                except Exception as e:
                    logger.error(f"{EMOJI['ERROR']} Error cleaning up {doc_id}: {e}")

            if deleted_count > 0:
                logger.info(f"{EMOJI['SUCCESS']} Cleaned up {deleted_count} orphaned signals")

            return deleted_count

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} cleanup_orphaned_active_signals failed: {e}")
            return 0

    def delete_active_signal(self, symbol: str) -> bool:
        """Delete active signal for a symbol."""
        if not self.is_available():
            return False

        try:
            active_col = self.db[self.active_collection]
            result = active_col.delete_one({"symbol": symbol})
            
            if result.deleted_count > 0:
                self.metrics["active_deletions"] += 1
                logger.info(f"{EMOJI['DELETE']} Deleted active signal for {symbol}")
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} delete_active_signal failed: {e}")
            return False

    def clear_all_active(self) -> int:
        """Clear all active signals."""
        if not self.is_available():
            return 0

        try:
            active_col = self.db[self.active_collection]
            result = active_col.delete_many({})
            self.metrics["active_deletions"] += result.deleted_count
            logger.info(f"{EMOJI['DB']} Cleared {result.deleted_count} active signals")
            return result.deleted_count

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} clear_all_active failed: {e}")
            return 0

    # ==================== UTILITY METHODS ====================

    def _generate_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for a signal."""
        symbol = data.get('symbol', 'UNKNOWN')
        signal_type = data.get('signal_type', 'UNKNOWN')
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        return f"{symbol}_{signal_type}_{timestamp}"

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'enabled': self.enabled,
            'connected': self._connected,
            'db_name': self.db_name,
            'version': self.version,
            'metrics': self.metrics,
        }

        if self.is_available():
            try:
                active_col = self.db[self.active_collection]
                resolved_col = self.db[self.resolved_collection]
                archive_col = self.db[self.archive_collection]

                stats['active_count'] = active_col.count_documents({})
                stats['resolved_count'] = resolved_col.count_documents({})
                stats['archive_count'] = archive_col.count_documents({})
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
