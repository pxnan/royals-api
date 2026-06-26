import csv
import logging
import os
import time

from flask import Blueprint, jsonify, request

from api import config, model_service
from api.database import get_db_connection, get_db_cursor

logger = logging.getLogger(__name__)
bp = Blueprint('debug', __name__)


@bp.route('/api/cek-csv', methods=['GET', 'OPTIONS'])
def cek_csv():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT pertanyaan, jawaban, kategori FROM dataset ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        header = ['pertanyaan', 'jawaban', 'kategori']
        data_rows = [[row['pertanyaan'], row['jawaban'], row['kategori']] for row in rows]
        all_rows = [header] + data_rows
        return jsonify({'total_lines': len(all_rows), 'total_rows': len(all_rows),
                        'last_5_rows': all_rows, 'file_path': 'database'}), 200
    except Exception as e:
        logger.error(f"Error in cek_csv: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/fix-csv', methods=['POST', 'OPTIONS'])
def fix_csv():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT pertanyaan, jawaban, kategori FROM dataset ORDER BY id")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
        with open(config.DATA_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['pertanyaan', 'jawaban', 'kategori'])
            for row in rows:
                writer.writerow([row['pertanyaan'], row['jawaban'], row['kategori']])
        return jsonify({'message': 'CSV berhasil diperbaiki dari database', 'total_rows': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/debug-db')
def debug_db():
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return "SUCCESS: Connected to MySQL"
        else:
            return "DB FAILED"
    except Exception as e:
        return f"FAILED: {str(e)}"


@bp.route('/api/test-db')
def test_db():
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return "DB OK"
        else:
            return "DB FAILED"
    except Exception as e:
        return str(e)


@bp.route('/api/debug-model', methods=['GET'])
def debug_model():
    """
    Endpoint diagnosa: cek koneksi Supabase, ukuran file model,
    koneksi DB, dan env variables — tanpa perlu lihat server log.
    Hapus endpoint ini setelah masalah terselesaikan.
    """
    result = {
        "env": {
            "SUPABASE_URL": config.SUPABASE_URL[:30] + "..." if config.SUPABASE_URL else None,
            "SUPABASE_ANON_KEY": "SET" if config.SUPABASE_ANON_KEY else "MISSING",
            "BUCKET_NAME": config.BUCKET_NAME,
            "MODEL_FILE": config.MODEL_FILE,
            "MYSQL_HOST": config.MYSQL_HOST[:20] + "..." if config.MYSQL_HOST else None,
        },
        "state": {
            "model_loaded": model_service.model_qa is not None,
            "vectorizer_loaded": model_service.vectorizer_qa is not None,
            "dataset_size": len(model_service.pertanyaan_list),
            "_models_loaded_flag": model_service._models_loaded,
            "_models_loading_flag": model_service._models_loading,
            "cache_built": model_service._X_all_questions_cache is not None,
        },
        "supabase_test": None,
        "supabase_file_size_bytes": None,
        "db_test": None,
        "errors": []
    }

    # Test Supabase
    if config.supabase and config.BUCKET_NAME and config.MODEL_FILE:
        try:
            t0 = time.time()
            data = config.supabase.storage.from_(config.BUCKET_NAME).download(config.MODEL_FILE)
            elapsed = round(time.time() - t0, 2)
            result["supabase_test"] = f"OK ({elapsed}s)"
            result["supabase_file_size_bytes"] = len(data) if data else 0
        except Exception as e:
            result["supabase_test"] = f"ERROR: {type(e).__name__}: {str(e)[:200]}"
            result["errors"].append(str(e))
    else:
        result["supabase_test"] = "SKIP - tidak terkonfigurasi"

    # Test DB
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dataset")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            result["db_test"] = f"OK - {count} rows in dataset"
        else:
            result["db_test"] = "FAILED - cannot connect"
    except Exception as e:
        result["db_test"] = f"ERROR: {str(e)[:200]}"
        result["errors"].append(str(e))

    return jsonify(result), 200


@bp.route('/api/debug-headers', methods=['GET'])
def debug_headers():
    return jsonify({
        "message": "Cek logs untuk detail",
        "origin": request.headers.get('Origin', 'Tidak ada origin'),
        "user_agent": request.headers.get('User-Agent', 'Tidak ada user-agent'),
        "host": request.headers.get('Host', 'Tidak ada host'),
        "x-forwarded-for": request.headers.get('X-Forwarded-For', 'Tidak ada info IP'),
    }), 200


