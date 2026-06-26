import logging
import threading

import mysql.connector
from mysql.connector import Error, pooling

from api import config

logger = logging.getLogger(__name__)

_db_pool = None
_db_pool_lock = threading.Lock()


def get_db_pool():
    global _db_pool
    if _db_pool is not None:
        return _db_pool
    with _db_pool_lock:
        if _db_pool is not None:
            return _db_pool
        try:
            _db_pool = pooling.MySQLConnectionPool(
                pool_name="chatbot_pool",
                pool_size=3,
                pool_reset_session=True,
                host=config.MYSQL_HOST,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                port=config.MYSQL_PORT,
                connection_timeout=10,
                connect_timeout=10,
            )
            logger.info("MySQL connection pool created (size=3)")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            _db_pool = None
    return _db_pool


def get_db_connection():
    """Ambil koneksi dari pool (fallback ke koneksi langsung jika pool gagal)."""
    pool = get_db_pool()
    if pool:
        try:
            return pool.get_connection()
        except Error as e:
            logger.warning(f"Pool exhausted or error, trying direct connect: {e}")
    try:
        conn = mysql.connector.connect(
            host=config.MYSQL_HOST,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            port=config.MYSQL_PORT,
            connection_timeout=10,
        )
        logger.info("MySQL direct connection successful (fallback)")
        return conn
    except Error as e:
        logger.error(f"MySQL connection error: {e}")
        return None


def get_db_cursor(conn, dictionary=True):
    if conn is None:
        return None
    return conn.cursor(dictionary=dictionary)
