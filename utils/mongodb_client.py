"""
MongoDB Client for Trading Bot - SUPER TDI + SUPER BOLLINGER BANDS STRATEGY.
ALIGNED: Super TDI + Super BB strategy with condition tracking.
Version: 3.4.5 - FIXED: Duplicate prevention with upsert, proper signal recovery
"""

import os
import logging
import json
import sys
import traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import quote_plus
import time

# Try to import numpy for type conversion
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# MongoDB imports
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError, OperationFailure, ConfigurationError
    MONGODB_AVAILABLE = True
except ImportError as e:
    MONGODB_AVAILABLE = False
    logging.warning(f"pymongo not available: {e}")

# Local imports
try:
    from settings import config
except ImportError:
    config = None

# Configure logging
logger = logging.getLogger(__name__)
mongo_logger = logging.getLogger("mongodb")

EMOJI = {
    "START": "🚀",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "WARNING": "⚠️",
    "INFO": "ℹ️",
    "DEBUG": "🔍",
    "DB": "💾",
    "SAVE": "💾",
    "DELETE": "🗑️",
    "UPDATE": "🔄",
    "FIND": "🔍",
    "INDEX": "📊",
    "AUTH": "🔐",
    "RAILWAY": "🚂",
    "ACTIVE": "🔴",
    "RESOLVED": "✅",
    "TDI": "📈",
    "BB": "📊",
    "CONDITION": "✅",
    "RETRY": "🔄",
    "STOP": "🛑",
    "FAILOVER": "💾",
    "CONVERT": "🔄",
    "FIXED": "🔧",
    "VERIFY": "✅",
    "RESTORE": "♻️",
}


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    Exposed for use by signal_manager.
    """
    if NUMPY_AVAILABLE and np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


class MongoDBClient:
    """
    MongoDB Client for Super TDI + Super Bollinger Bands strategy.
    VERSION 3.4.5: Fixed duplicate prevention with upsert.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self.initialized = False
        self.version = "3.4.5"
        self._connected = False
        self._last_error = None
        self._connection_attempts = 0
        self._failover_dir = "mongodb_failover"
        self._last_save_attempt = None
        self._last_save_success = False

        # Collection names
        self.ACTIVE_COLLECTION = "active_signals"
        self.RESOLVED_COLLECTION = "resolved_signals"
        self.SIGNALS_COLLECTION = "signals"
        self.STATS_COLLECTION = "bot_stats"
        self.ERRORS_COLLECTION = "errors"

        # Metrics
        self.metrics = {
            "active_deletions": 0,
            "active_updates": 0,
            "resolved_saves": 0,
            "archive_saves": 0,
            "errors": 0,
            "signals_saved": 0,
            "failover_saves": 0,
            "connection_errors": 0,
            "type_conversions": 0,
            "save_attempts": 0,
            "save_failures": 0,
            "upsert_saves": 0,
        }

        # Create failover directory
        try:
            os.makedirs(self._failover_dir, exist_ok=True)
            mongo_logger.info(f"{EMOJI['FAILOVER']} Failover directory: {self._failover_dir}")
        except Exception as e:
            mongo_logger.warning(f"{EMOJI['WARNING']} Could not create failover directory: {e}")

        self._initialize()

    # ==================== TYPE CONVERSION ====================

    def _convert_to_serializable(self, data: Any, depth: int = 0) -> Any:
        """Convert ALL non-serializable types to Python native types."""
        if depth > 20:
            return str(data)

        if data is None:
            return None

        # Handle NumPy types
        if NUMPY_AVAILABLE and np is not None:
            try:
                if isinstance(data, np.bool_):
                    self.metrics["type_conversions"] += 1
                    return bool(data)
                elif isinstance(data, np.integer):
                    self.metrics["type_conversions"] += 1
                    return int(data)
                elif isinstance(data, np.floating):
                    self.metrics["type_conversions"] += 1
                    return float(data)
                elif isinstance(data, np.ndarray):
                    self.metrics["type_conversions"] += 1
                    return data.tolist()
                elif isinstance(data, np.generic):
                    self.metrics["type_conversions"] += 1
                    return data.item()
            except Exception as e:
                mongo_logger.debug(f"NumPy conversion warning: {e}")

        # Handle datetime
        if isinstance(data, datetime):
            return data.isoformat()

        # Handle dict - RECURSIVE
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v, depth + 1) for k, v in data.items()}

        # Handle list or tuple - RECURSIVE
        if isinstance(data, (list, tuple)):
            return [self._convert_to_serializable(item, depth + 1) for item in data]

        # Handle primitive types
        if isinstance(data, (bool, int, float, str)):
            return data

        # Handle bytes
        if isinstance(data, bytes):
            try:
                return data.decode('utf-8')
            except:
                return str(data)

        # Handle sets
        if isinstance(data, set):
            return list(data)

        # Fallback
        try:
            return str(data)
        except:
            return f"<Unserializable: {type(data).__name__}>"

    def _prepare_for_mongodb(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for MongoDB by converting ALL types."""
        try:
            return self._convert_to_serializable(data)
        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} Type conversion failed: {e}")
            try:
                return json.loads(json.dumps(data, default=str))
            except:
                return {"_error": f"Type conversion failed: {str(data)[:100]}"}

    # ==================== SIGNAL OPERATIONS ====================

    def get_active_signal_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get active signal by symbol - checks active collection.
        RETURNS: Signal dict or None
        """
        if not self.is_available():
            return None

        try:
            active_collection = self.db[self.ACTIVE_COLLECTION]
            signal = active_collection.find_one({
                'symbol': symbol,
                'status': {'$in': ['ACTIVE', 'AI_WAITING', 'EXIT_1H']}
            })
            
            if signal:
                signal['_id'] = str(signal['_id'])
                mongo_logger.debug(f"Found active signal for {symbol}: {signal['_id']}")
                return signal
            return None

        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} get_active_signal_by_symbol failed: {e}")
            return None

    def save_signal_upsert(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Save signal using upsert to prevent duplicate key errors.
        This is the PRIMARY method for saving signals.
        RETURNS: doc_id or None
        """
        self.metrics["save_attempts"] += 1
        self._last_save_attempt = datetime.now().isoformat()
        
        if not self.is_available():
            mongo_logger.warning(f"{EMOJI['WARNING']} MongoDB not available - saving to failover")
            return self._save_to_failover(signal_data)

        try:
            # Convert types
            signal_data = self._convert_to_serializable(signal_data)
            
            # Add metadata
            signal_data['created_at'] = datetime.now().isoformat()
            signal_data['updated_at'] = datetime.now().isoformat()
            signal_data['strategy_version'] = self.version
            signal_data['strategy_name'] = "Super TDI + Super Bollinger Bands"
            
            # Generate ID if not present
            if '_id' not in signal_data or not signal_data['_id']:
                signal_data['_id'] = self._generate_id(signal_data)
            
            doc_id = signal_data['_id']
            symbol = signal_data.get('symbol')
            
            if not symbol:
                mongo_logger.warning("No symbol in signal data")
                return None
            
            active_collection = self.db[self.ACTIVE_COLLECTION]
            
            mongo_logger.info(f"{EMOJI['SAVE']} Saving signal for {symbol} using upsert...")
            mongo_logger.debug(f"  doc_id: {doc_id}")
            mongo_logger.debug(f"  conditions: {signal_data.get('conditions_met', 0)}/{signal_data.get('conditions_total', 5)}")
            
            # ===== USE UPSERT TO PREVENT DUPLICATE KEY ERRORS =====
            result = active_collection.update_one(
                {'symbol': symbol, 'status': {'$in': ['ACTIVE', 'AI_WAITING']}},
                {'$set': signal_data},
                upsert=True
            )

            if result.acknowledged:
                self.metrics["signals_saved"] += 1
                self.metrics["upsert_saves"] += 1
                self._last_save_success = True
                
                if result.upserted_id:
                    mongo_logger.info(f"{EMOJI['SUCCESS']} ✅ New signal saved: {doc_id} for {symbol}")
                else:
                    mongo_logger.info(f"{EMOJI['UPDATE']} ✅ Existing signal updated: {doc_id} for {symbol}")
                
                # Also save to archive
                try:
                    signals_collection = self.db[self.SIGNALS_COLLECTION]
                    signals_collection.update_one(
                        {'_id': doc_id},
                        {'$set': signal_data},
                        upsert=True
                    )
                    self.metrics["archive_saves"] += 1
                except Exception as e:
                    mongo_logger.debug(f"Archive save failed: {e}")
                
                # Save to failover as backup
                self._save_to_failover(signal_data)
                
                return doc_id
            else:
                mongo_logger.warning(f"Save not acknowledged for {symbol}")
                return None

        except DuplicateKeyError as e:
            mongo_logger.warning(f"{EMOJI['WARNING']} Duplicate key: {e}")
            # Try to recover by getting existing signal
            try:
                symbol = signal_data.get('symbol')
                if symbol:
                    existing = self.get_active_signal_by_symbol(symbol)
                    if existing:
                        mongo_logger.info(f"{EMOJI['SUCCESS']} Found existing signal: {existing.get('_id')}")
                        return str(existing['_id'])
            except:
                pass
            return self._save_to_failover(signal_data)
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.metrics["save_failures"] += 1
            mongo_logger.error(f"{EMOJI['ERROR']} save_signal_upsert failed: {e}")
            mongo_logger.error(f"  Traceback: {traceback.format_exc()}")
            return self._save_to_failover(signal_data)

    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Legacy save method - now calls save_signal_upsert.
        Kept for backward compatibility.
        """
        return self.save_signal_upsert(signal_data)

    def _save_to_failover(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """Save signal to local file as failover."""
        try:
            serializable_data = self._convert_to_serializable(signal_data)
            
            doc_id = serializable_data.get('_id') or self._generate_id(serializable_data)
            serializable_data['_id'] = doc_id
            serializable_data['failover_saved_at'] = datetime.now().isoformat()
            
            # Save to individual file
            filename = f"{self._failover_dir}/{doc_id}.json"
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, default=str, indent=2)
            
            # Also append to master failover file
            master_file = f"{self._failover_dir}/failover_archive.json"
            try:
                with open(master_file, 'r') as f:
                    archive = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                archive = []
            
            existing = [s for s in archive if s.get('_id') == doc_id]
            if not existing:
                archive.append(serializable_data)
                with open(master_file, 'w') as f:
                    json.dump(archive, f, default=str, indent=2)
            
            self.metrics["failover_saves"] += 1
            return doc_id
        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} Failover save failed: {e}")
            return None

    def _get_from_failover(self, doc_id: str) -> Optional[Dict]:
        """Get a signal from failover files."""
        try:
            filename = f"{self._failover_dir}/{doc_id}.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
            return None
        except:
            return None

    def get_active_signals(self, limit: int = 100) -> Dict[str, Dict]:
        """Get all active signals from ACTIVE collection."""
        if not self.is_available():
            return self._get_failover_active_signals()

        try:
            active_collection = self.db[self.ACTIVE_COLLECTION]
            signals = active_collection.find(
                {'status': {'$ne': 'RESOLVED'}},
                limit=limit
            ).sort('entry_time', DESCENDING)

            result = {}
            for signal in signals:
                doc_id = str(signal['_id'])
                signal['_id'] = doc_id
                result[doc_id] = signal

            mongo_logger.debug(f"{EMOJI['ACTIVE']} Found {len(result)} active signals")
            return result

        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} get_active_signals failed: {e}")
            return self._get_failover_active_signals()

    def _get_failover_active_signals(self) -> Dict[str, Dict]:
        """Get active signals from failover files."""
        result = {}
        try:
            master_file = f"{self._failover_dir}/failover_archive.json"
            if os.path.exists(master_file):
                with open(master_file, 'r') as f:
                    archive = json.load(f)
                for signal in archive:
                    status = signal.get('status', 'ACTIVE')
                    if status != 'RESOLVED' and status != 'CLOSED':
                        doc_id = signal.get('_id', 'unknown')
                        result[doc_id] = signal
            return result
        except:
            return {}

    def move_to_resolved(self, doc_id: str, signal_data: Dict[str, Any]) -> bool:
        """
        Move a signal to resolved collection and delete from active.
        COMPREHENSIVE: Handles all data fields properly.
        """
        # Convert types
        signal_data = self._convert_to_serializable(signal_data)
        
        mongo_logger.info(f"{EMOJI['RESOLVED']} Moving {doc_id} to resolved...")
        
        # Save to failover first
        self._save_to_failover(signal_data)

        if not self.is_available():
            mongo_logger.warning(f"{EMOJI['WARNING']} MongoDB not available - saved to failover")
            return True

        try:
            if '_id' not in signal_data:
                signal_data['_id'] = doc_id

            signal_data['resolved_at'] = datetime.now().isoformat()
            signal_data['strategy_version'] = self.version
            signal_data['strategy_name'] = "Super TDI + Super Bollinger Bands"
            signal_data['original_doc_id'] = doc_id

            # ===== STEP 1: SAVE TO RESOLVED COLLECTION =====
            resolved_collection = self.db[self.RESOLVED_COLLECTION]
            
            mongo_logger.info(f"{EMOJI['DB']} Saving to {self.RESOLVED_COLLECTION}: {doc_id}")
            mongo_logger.debug(f"  Conditions: {signal_data.get('conditions_met', 'N/A')}/{signal_data.get('conditions_total', 5)}")
            mongo_logger.debug(f"  PnL: ${signal_data.get('pnl', 0):.2f}")
            
            resolved_result = resolved_collection.update_one(
                {'_id': doc_id},
                {'$set': signal_data},
                upsert=True
            )

            if resolved_result.acknowledged:
                self.metrics["resolved_saves"] += 1
                mongo_logger.info(f"{EMOJI['RESOLVED']} ✅ Saved to resolved: {doc_id}")
            else:
                mongo_logger.warning(f"{EMOJI['WARNING']} Failed to save {doc_id} to resolved")

            # ===== STEP 2: DELETE FROM ACTIVE COLLECTION =====
            active_collection = self.db[self.ACTIVE_COLLECTION]
            delete_result = active_collection.delete_one({'_id': doc_id})

            if delete_result.deleted_count > 0:
                self.metrics["active_deletions"] += 1
                mongo_logger.info(f"{EMOJI['DELETE']} ✅ Deleted from active: {doc_id}")
            else:
                mongo_logger.warning(f"{EMOJI['WARNING']} Failed to delete {doc_id} from active")

            # ===== STEP 3: UPDATE ARCHIVE =====
            try:
                signals_collection = self.db[self.SIGNALS_COLLECTION]
                signals_collection.update_one(
                    {'_id': doc_id},
                    {'$set': signal_data},
                    upsert=True
                )
                self.metrics["archive_saves"] += 1
                mongo_logger.debug(f"{EMOJI['DB']} Archived: {doc_id}")
            except Exception as e:
                mongo_logger.warning(f"{EMOJI['WARNING']} Failed to archive: {e}")

            # ===== STEP 4: DELETE FAILOVER =====
            self._delete_failover_file(doc_id)

            return True

        except Exception as e:
            self.metrics["errors"] += 1
            mongo_logger.error(f"{EMOJI['ERROR']} move_to_resolved failed: {e}")
            mongo_logger.error(f"  Traceback: {traceback.format_exc()}")
            return True  # Return True to continue trading

    def update_signal_status(self, doc_id: str, status: str, update_data: Dict[str, Any]) -> bool:
        """
        Update signal status - calls move_to_resolved for terminal statuses.
        """
        # Convert types
        update_data = self._convert_to_serializable(update_data)
        update_data['status'] = status
        
        terminal_statuses = ['PROFIT', 'LOSS', 'BREAK_EVEN', 'CLOSED', 'EXIT_1H', 'EXIT_TARGET']
        
        if status in terminal_statuses:
            return self.move_to_resolved(doc_id, update_data)
        else:
            # Non-terminal: update in active
            if not self.is_available():
                self._save_to_failover(update_data)
                return True
            
            try:
                active_collection = self.db[self.ACTIVE_COLLECTION]
                result = active_collection.update_one(
                    {'_id': doc_id},
                    {'$set': update_data}
                )
                return True
            except Exception as e:
                mongo_logger.error(f"{EMOJI['ERROR']} update_signal_status failed: {e}")
                return True

    def cleanup_orphaned_active_signals(self) -> int:
        """Clean up orphaned active signals that should have been deleted."""
        if not self.is_available():
            return 0

        try:
            active_collection = self.db[self.ACTIVE_COLLECTION]
            terminal_statuses = ['PROFIT', 'LOSS', 'BREAK_EVEN', 'CLOSED', 'RESOLVED', 'EXIT_1H', 'EXIT_TARGET']

            orphaned = active_collection.find({'status': {'$in': terminal_statuses}})
            orphaned_list = list(orphaned)
            deleted_count = 0

            for signal in orphaned_list:
                doc_id = signal.get('_id')
                if doc_id:
                    result = active_collection.delete_one({'_id': doc_id})
                    if result.deleted_count > 0:
                        deleted_count += 1
                        self.metrics["active_deletions"] += 1
                        mongo_logger.info(f"{EMOJI['DELETE']} Cleaned up orphaned signal: {doc_id}")

            if deleted_count > 0:
                mongo_logger.info(f"{EMOJI['SUCCESS']} Cleaned up {deleted_count} orphaned signals")

            return deleted_count

        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} cleanup_orphaned_active_signals failed: {e}")
            return 0

    def _delete_failover_file(self, doc_id: str):
        """Delete a failover file after successful recovery."""
        try:
            if doc_id:
                filename = f"{self._failover_dir}/{doc_id}.json"
                if os.path.exists(filename):
                    os.remove(filename)
        except Exception as e:
            mongo_logger.debug(f"Could not delete failover file: {e}")

    # ==================== INITIALIZATION ====================

    def _initialize(self):
        """Initialize MongoDB connection."""
        try:
            mongo_logger.info(f"{EMOJI['START']} MONGODB_INIT: START v{self.version}")
            mongo_logger.info(f"{EMOJI['RAILWAY']} MONGODB_INIT: Running on Railway")

            self.mongo_uri = self._get_mongo_uri()

            if not self.mongo_uri:
                mongo_logger.warning(f"{EMOJI['WARNING']} MONGODB_INIT: No MongoDB URI found. Running in failover-only mode.")
                self.initialized = False
                return

            self.db_name = self._get_db_name()

            redacted_uri = self._redact_uri(self.mongo_uri)
            mongo_logger.info(f"{EMOJI['INFO']} MONGODB_INIT: Connecting to {redacted_uri}")
            mongo_logger.info(f"{EMOJI['INFO']} MONGODB_INIT: Database: {self.db_name}")

            self._connect()
            self._create_indexes()

            self.initialized = True
            self._connected = True

            mongo_logger.info(f"{EMOJI['SUCCESS']} MONGODB_INIT: Connected to MongoDB")
            mongo_logger.info(f"{EMOJI['DB']} Database: {self.db_name}")
            mongo_logger.info(f"{EMOJI['SUCCESS']} MONGODB_INIT v{self.version}: Client initialized")
            mongo_logger.info(f"{EMOJI['DELETE']} Active signals will be COMPLETELY DELETED when resolved")
            mongo_logger.info(f"{EMOJI['FAILOVER']} Failover directory: {self._failover_dir}")
            mongo_logger.info(f"{EMOJI['CONVERT']} NumPy type conversion: {'✅' if NUMPY_AVAILABLE else '❌'}")

        except OperationFailure as e:
            error_msg = str(e)
            mongo_logger.error(f"{EMOJI['ERROR']} MONGODB_INIT: Authentication failed - {error_msg}")
            self.initialized = False
            self._connected = False
            self._last_error = error_msg

        except ConnectionFailure as e:
            mongo_logger.error(f"{EMOJI['ERROR']} MONGODB_INIT: Connection failed - {e}")
            self.initialized = False
            self._connected = False
            self._last_error = str(e)

        except ServerSelectionTimeoutError as e:
            mongo_logger.error(f"{EMOJI['ERROR']} MONGODB_INIT: Server selection timeout - {e}")
            self.initialized = False
            self._connected = False
            self._last_error = str(e)

        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} MONGODB_INIT: Failed - {e}")
            self.initialized = False
            self._connected = False
            self._last_error = str(e)

    def _redact_uri(self, uri: str) -> str:
        """Redact sensitive parts of URI for logging."""
        if not uri:
            return "None"
        import re
        redacted = re.sub(r'://[^@]+@', '://***:***@', uri)
        return redacted

    def _get_mongo_uri(self) -> str:
        """Get MongoDB URI from environment or config."""
        mongo_uri = None

        env_vars = [
            'MONGODB_URI',
            'MONGODB_URL',
            'MONGO_URI',
            'MONGO_URL',
            'RAILWAY_MONGODB_URI',
            'RAILWAY_MONGODB_URL'
        ]

        for var in env_vars:
            value = os.environ.get(var)
            if value:
                mongo_uri = value
                mongo_logger.debug(f"{EMOJI['DEBUG']} Found {var}")
                break

        if not mongo_uri:
            try:
                mongo_uri = getattr(config, 'mongodb_uri', None)
                if mongo_uri:
                    mongo_logger.debug(f"{EMOJI['DEBUG']} Found in config.mongodb_uri")
            except:
                pass

        if not mongo_uri:
            mongo_uri = self._build_uri_from_components()

        if mongo_uri:
            mongo_logger.info(f"{EMOJI['INFO']} MONGODB URI found: {self._redact_uri(mongo_uri)}")
        else:
            mongo_logger.warning(f"{EMOJI['WARNING']} MONGODB URI not found")

        return mongo_uri

    def _build_uri_from_components(self) -> Optional[str]:
        """Build MongoDB URI from individual components."""
        try:
            host = os.environ.get('MONGODB_HOST', os.environ.get('MONGO_HOST', 'mongodb.railway.internal'))
            port = os.environ.get('MONGODB_PORT', os.environ.get('MONGO_PORT', '27017'))
            user = os.environ.get('MONGODB_USER', os.environ.get('MONGO_USER'))
            password = os.environ.get('MONGODB_PASSWORD', os.environ.get('MONGO_PASS'))
            db_name = self._get_db_name()

            if not host:
                return None

            if 'railway.internal' in host:
                mongo_logger.info(f"{EMOJI['RAILWAY']} Using Railway internal host: {host}")

            if user and password:
                encoded_user = quote_plus(user)
                encoded_password = quote_plus(password)

                if '.mongodb.net' in host:
                    return f"mongodb+srv://{encoded_user}:{encoded_password}@{host}/{db_name}?retryWrites=true&w=majority"
                else:
                    return f"mongodb://{encoded_user}:{encoded_password}@{host}:{port}/{db_name}?authSource=admin&retryWrites=true&w=majority"
            elif user:
                return f"mongodb://{quote_plus(user)}@{host}:{port}/{db_name}?authSource=admin"
            else:
                return f"mongodb://{host}:{port}/{db_name}"

        except Exception as e:
            mongo_logger.warning(f"{EMOJI['WARNING']} MONGODB_INIT: Failed to build URI from components: {e}")
            return None

    def _get_db_name(self) -> str:
        """Get database name from environment or config."""
        db_name = os.environ.get('MONGODB_DB', os.environ.get('MONGO_DB'))

        if not db_name:
            try:
                db_name = getattr(config, 'mongodb_db_name', 'trading_bot')
            except:
                db_name = 'trading_bot'

        return db_name

    def _connect(self):
        """Connect to MongoDB with retry logic."""
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not available")

        max_retries = 5
        retry_delay = 3

        mongo_logger.info(f"{EMOJI['INFO']} Attempting to connect to MongoDB (max {max_retries} retries)...")

        for attempt in range(max_retries):
            try:
                self._connection_attempts += 1

                if 'railway.internal' in self.mongo_uri:
                    mongo_logger.info(f"{EMOJI['RAILWAY']} Connecting to Railway internal MongoDB...")

                self.client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=15000,
                    connectTimeoutMS=15000,
                    socketTimeoutMS=15000,
                    maxPoolSize=50,
                    minPoolSize=5,
                    retryWrites=True,
                    w='majority',
                    tlsAllowInvalidCertificates=True,
                    tlsAllowInvalidHostnames=True,
                )

                self.client.admin.command('ping')

                self.db = self.client[self.db_name]

                collections = self.db.list_collection_names()
                mongo_logger.info(f"{EMOJI['DEBUG']} Available collections: {collections}")

                mongo_logger.info(f"{EMOJI['SUCCESS']} Connected to MongoDB on attempt {attempt + 1}")
                return

            except OperationFailure as e:
                error_msg = str(e)
                if 'Authentication failed' in error_msg or 'auth failed' in error_msg.lower():
                    mongo_logger.error(f"{EMOJI['AUTH']} Authentication failed - check username/password")
                    raise
                elif attempt < max_retries - 1:
                    mongo_logger.warning(f"{EMOJI['RETRY']} Operation failed, retrying in {retry_delay}s: {e}")
                    time.sleep(retry_delay)
                else:
                    raise

            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    mongo_logger.warning(f"{EMOJI['RETRY']} Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    mongo_logger.error(f"{EMOJI['ERROR']} All connection attempts failed")
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    mongo_logger.warning(f"{EMOJI['RETRY']} Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise

    def _create_indexes(self):
        """Create necessary indexes for performance."""
        try:
            if self.db is None:
                return

            # Active signals indexes
            active_collection = self.db[self.ACTIVE_COLLECTION]
            active_collection.create_index([("symbol", ASCENDING)])
            active_collection.create_index([("status", ASCENDING)])
            active_collection.create_index([("entry_time", DESCENDING)])
            active_collection.create_index([("symbol", ASCENDING), ("status", ASCENDING)])

            # Resolved signals indexes
            resolved_collection = self.db[self.RESOLVED_COLLECTION]
            resolved_collection.create_index([("symbol", ASCENDING)])
            resolved_collection.create_index([("status", ASCENDING)])
            resolved_collection.create_index([("exit_time", DESCENDING)])

            # Signals archive indexes
            signals_collection = self.db[self.SIGNALS_COLLECTION]
            signals_collection.create_index([("symbol", ASCENDING), ("signal_type", ASCENDING)])
            signals_collection.create_index([("timestamp", DESCENDING)])

            mongo_logger.info(f"{EMOJI['INDEX']} MONGODB: Indexes created")

        except OperationFailure as e:
            mongo_logger.warning(f"{EMOJI['WARNING']} MONGODB: Index creation failed - {e}")
        except Exception as e:
            mongo_logger.warning(f"{EMOJI['WARNING']} MONGODB: Index creation failed - {e}")

    # ==================== UTILITY METHODS ====================

    def is_available(self) -> bool:
        """Check if MongoDB is available."""
        if not self._connected or self.client is None or self.db is None:
            return False

        try:
            self.client.admin.command('ping')
            return True
        except:
            self._connected = False
            return False

    def is_initialized(self) -> bool:
        """Check if MongoDB is initialized."""
        return self.initialized and self.is_available()

    def _generate_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for a signal."""
        symbol = data.get('symbol', 'UNKNOWN')
        signal_type = data.get('signal_type', 'UNKNOWN')
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        return f"{symbol}_{signal_type}_{timestamp}"

    def get_collection(self, collection_name: str):
        """Get a collection by name."""
        if not self.is_available():
            return None
        return self.db[collection_name]

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        if not self.is_available():
            return {}

        try:
            collection = self.db[self.STATS_COLLECTION]
            stats = collection.find_one({'_id': 'bot_stats'})
            if stats:
                stats['_id'] = str(stats['_id'])
                return stats
            return {}
        except Exception as e:
            mongo_logger.error(f"{EMOJI['ERROR']} get_stats failed: {e}")
            return {}

    def cleanup(self):
        """Clean up MongoDB connection."""
        if self.client:
            self.client.close()
            mongo_logger.info(f"{EMOJI['STOP']} MONGODB: Connection closed")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.cleanup()


# Create singleton instance
mongodb_client = MongoDBClient()

# Export for compatibility
firebase_client = mongodb_client

__all__ = [
    'mongodb_client',
    'firebase_client',
    'MongoDBClient',
    'convert_numpy_types',
]
