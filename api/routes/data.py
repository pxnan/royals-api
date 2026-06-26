import csv
import io
import logging
import time

from flask import Blueprint, Response, jsonify, request

from preprocessing import preprocess

from api import model_service
from api.auth import api_key_required
from api.database import get_db_connection, get_db_cursor

logger = logging.getLogger(__name__)
bp = Blueprint('data', __name__)


@bp.route('/api/pertanyaan-unknown', methods=['GET', 'OPTIONS'])
@api_key_required
def get_unknown_questions():
    if request.method == 'OPTIONS':
        return '', 200
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    conn = get_db_connection()
    if conn is None:
        return jsonify({'data': [], 'total_data': 0, 'page': page, 'per_page': per_page, 'total_pages': 1}), 200
    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("SELECT * FROM pertanyaan_unknow ORDER BY id DESC LIMIT %s OFFSET %s", (per_page, offset))
    data = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) as total FROM pertanyaan_unknow")
    total = cursor.fetchone()['total']
    cursor.close()
    conn.close()
    total_pages = (total + per_page - 1) // per_page
    return jsonify({'page': page, 'per_page': per_page, 'total_data': total, 'total_pages': total_pages, 'data': data}), 200


@bp.route('/api/delete-unknown', methods=['DELETE', 'OPTIONS'])
@api_key_required
def delete_unknown():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}
    unknown_id = data.get('id')
    if not unknown_id:
        return jsonify({'error': 'ID tidak ditemukan'}), 400
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Pertanyaan berhasil dihapus (demo)'}), 200
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pertanyaan_unknow WHERE id = %s", (unknown_id,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    if affected == 0:
        return jsonify({'error': 'Data tidak ditemukan'}), 404
    return jsonify({'message': 'Pertanyaan berhasil dihapus', 'status': 'success'}), 200


@bp.route('/api/delete-all-unknown', methods=['DELETE', 'OPTIONS'])
@api_key_required
def delete_all_unknown():
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Semua pertanyaan berhasil dihapus (demo)', 'deleted_count': 0}), 200
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pertanyaan_unknow")
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return jsonify({'message': f'{affected} pertanyaan berhasil dihapus', 'status': 'success', 'deleted_count': affected}), 200


# ==================== KATEGORI & MODEL INFO ====================
@bp.route('/api/kategori', methods=['GET', 'OPTIONS'])
@api_key_required
def get_kategori():
    if request.method == 'OPTIONS':
        return '', 200
    model_service.load_models_and_data()
    categories = sorted(list(set(model_service.kategori_list)))
    return jsonify({'kategori': categories})


@bp.route('/api/model-info', methods=['GET', 'OPTIONS'])
@api_key_required
def model_info():
    if request.method == 'OPTIONS':
        return '', 200
    model_service.load_models_and_data()
    return jsonify({
        'total_questions': len(model_service.pertanyaan_list),
        'total_answers': len(model_service.answers),
        'categories': sorted(list(set(model_service.kategori_list))),
        'model_loaded': model_service.model_qa is not None,
        'vectorizer_loaded': model_service.vectorizer_qa is not None
    }), 200


# ==================== DATASET MANAGEMENT ====================
@bp.route('/api/get-all-data', methods=['GET', 'OPTIONS'])
@api_key_required
def get_all_data():
    if request.method == 'OPTIONS':
        return '', 200
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    search = request.args.get('search', '', type=str)
    kategori_filter = request.args.get('kategori', '', type=str)
    offset = (page - 1) * per_page

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = get_db_cursor(conn, dictionary=True)
        base_query = "SELECT id, pertanyaan, jawaban, kategori FROM dataset WHERE 1=1"
        params = []

        if search:
            base_query += " AND (LOWER(pertanyaan) LIKE LOWER(%s) OR LOWER(jawaban) LIKE LOWER(%s))"
            search_param = f"%{search}%"
            params.extend([search_param, search_param])

        if kategori_filter:
            base_query += " AND kategori = %s"
            params.append(kategori_filter)

        count_query = f"SELECT COUNT(*) as total FROM ({base_query}) as sub"
        cursor.execute(count_query, params)
        total = cursor.fetchone()['total']
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        query = base_query + " ORDER BY id LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()

        data = [{'index': row['id'] - 1, 'id': row['id'], 'pertanyaan': row['pertanyaan'],
                 'jawaban': row['jawaban'], 'kategori': row['kategori']} for row in rows]

        cursor.execute("SELECT DISTINCT kategori FROM dataset ORDER BY kategori")
        categories = [row['kategori'] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return jsonify({'page': page, 'per_page': per_page, 'total_data': total,
                        'total_pages': total_pages, 'data': data, 'categories': categories}), 200
    except Exception as e:
        logger.error(f"Error in get_all_data: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/unknown/recent', methods=['GET', 'OPTIONS'])
@api_key_required
def get_recent_unknown():
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'data': [], 'total': 0}), 200
    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT id, pertanyaan, created_at FROM pertanyaan_unknow ORDER BY id DESC LIMIT 5")
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({'data': data, 'total': len(data)}), 200
    except Exception as e:
        logger.error(f"Error in get_recent_unknown: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/dataset/recent', methods=['GET', 'OPTIONS'])
@api_key_required
def get_recent_dataset():
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'data': [], 'total': 0}), 500
    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT id, pertanyaan, jawaban, kategori FROM dataset ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        data = [{
            'id': row['id'],
            'pertanyaan': row['pertanyaan'],
            'jawaban': row['jawaban'][:50] + ('...' if len(row['jawaban']) > 50 else '') if row['jawaban'] else '',
            'jawaban_full': row['jawaban'],
            'kategori': row['kategori']
        } for row in rows]
        return jsonify({'data': data, 'total': len(data)}), 200
    except Exception as e:
        logger.error(f"Error in get_recent_dataset: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/tambah-data', methods=['POST', 'OPTIONS'])
@api_key_required
def tambah_data():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}
    pertanyaan = data.get('pertanyaan', '').strip()
    jawaban = data.get('jawaban', '').strip()
    kategori = data.get('kategori', '').strip()
    if not pertanyaan or not jawaban or not kategori:
        return jsonify({'error': 'Semua field harus diisi'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM dataset WHERE LOWER(pertanyaan) = LOWER(%s)", (pertanyaan,))
        if cursor.fetchone():
            return jsonify({'error': f'Pertanyaan "{pertanyaan}" sudah ada', 'status': 'duplicate'}), 409
        cursor.execute("INSERT INTO dataset (pertanyaan, jawaban, kategori) VALUES (%s, %s, %s)",
                       (pertanyaan, jawaban, kategori))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({
            'message': 'Data berhasil ditambahkan',
            'data': {'pertanyaan': pertanyaan, 'jawaban': jawaban, 'kategori': kategori},
            'status': 'success'
        }), 201
    except Exception as e:
        conn.rollback()
        logger.error(f"Error in tambah_data: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/update-data', methods=['PUT', 'OPTIONS'])
@api_key_required
def update_data():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}

    item_id = data.get('id')
    index = None
    if item_id is None:
        index = data.get('index')
        if index is None:
            return jsonify({'error': 'Parameter "id" atau "index" diperlukan'}), 400

    pertanyaan = data.get('pertanyaan', '').strip()
    jawaban = data.get('jawaban', '').strip()
    kategori = data.get('kategori', '').strip()
    if not pertanyaan or not jawaban or not kategori:
        return jsonify({'error': 'Semua field harus diisi'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        if item_id is not None:
            cursor = conn.cursor()
            cursor.execute("UPDATE dataset SET pertanyaan=%s, jawaban=%s, kategori=%s WHERE id=%s",
                           (pertanyaan, jawaban, kategori, item_id))
            if cursor.rowcount == 0:
                cursor.close()
                conn.close()
                return jsonify({'error': f'Data dengan ID {item_id} tidak ditemukan'}), 404
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'message': 'Data berhasil diupdate', 'status': 'success'}), 200

        if index is not None:
            cursor = get_db_cursor(conn, dictionary=True)
            cursor.execute("SELECT id FROM dataset ORDER BY id")
            ids = [row['id'] for row in cursor.fetchall()]
            cursor.close()
            if index < 0 or index >= len(ids):
                return jsonify({'error': f'Index {index} tidak valid (0-{len(ids)-1})'}), 400
            target_id = ids[index]
            cursor = conn.cursor()
            cursor.execute("UPDATE dataset SET pertanyaan=%s, jawaban=%s, kategori=%s WHERE id=%s",
                           (pertanyaan, jawaban, kategori, target_id))
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'message': 'Data berhasil diupdate', 'status': 'success'}), 200
    except Exception as e:
        conn.rollback()
        logger.error(f"Error in update_data: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/delete-data', methods=['DELETE', 'OPTIONS'])
@api_key_required
def delete_data():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    item_id = data.get('id')
    if item_id is None:
        index = data.get('index')
        if index is not None:
            conn = get_db_connection()
            if not conn:
                return jsonify({'error': 'Database tidak tersedia'}), 500
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM dataset ORDER BY id")
            ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            try:
                idx = int(index)
                if 0 <= idx < len(ids):
                    item_id = ids[idx]
                else:
                    return jsonify({'error': f'Index {idx} tidak valid (0-{len(ids)-1})'}), 400
            except Exception:
                return jsonify({'error': 'Index harus angka'}), 400
        else:
            return jsonify({'error': 'Parameter "id" atau "index" diperlukan'}), 400

    try:
        item_id = int(item_id)
    except Exception:
        return jsonify({'error': 'ID harus angka'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM dataset WHERE id = %s", (item_id,))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        conn.close()
        if affected == 0:
            return jsonify({'error': f'Data dengan ID {item_id} tidak ditemukan'}), 404
        return jsonify({'message': 'Data berhasil dihapus', 'status': 'success', 'deleted_id': item_id}), 200
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/delete-bulk-data', methods=['DELETE', 'OPTIONS'])
@api_key_required
def delete_bulk_data():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}
    ids = data.get('ids')
    indices = data.get('indices')

    if ids is not None and isinstance(ids, list) and len(ids) > 0:
        try:
            ids = [int(i) for i in ids]
        except Exception:
            return jsonify({'error': 'ID harus berupa angka'}), 400
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        try:
            cursor = conn.cursor()
            placeholders = ','.join(['%s'] * len(ids))
            cursor.execute(f"DELETE FROM dataset WHERE id IN ({placeholders})", ids)
            conn.commit()
            deleted = cursor.rowcount
            cursor.close()
            conn.close()
            return jsonify({'message': f'{deleted} data berhasil dihapus', 'status': 'success'}), 200
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in delete_bulk_data (ids): {e}")
            if conn:
                conn.close()
            return jsonify({'error': str(e)}), 500

    if indices is not None and isinstance(indices, list) and len(indices) > 0:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        try:
            cursor = get_db_cursor(conn, dictionary=True)
            cursor.execute("SELECT id FROM dataset ORDER BY id")
            ids_all = [row['id'] for row in cursor.fetchall()]
            cursor.close()
            to_delete = []
            for idx in indices:
                try:
                    i = int(idx)
                    if 0 <= i < len(ids_all):
                        to_delete.append(ids_all[i])
                except Exception:
                    pass
            if not to_delete:
                return jsonify({'error': 'Tidak ada data valid yang dipilih'}), 400
            placeholders = ','.join(['%s'] * len(to_delete))
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM dataset WHERE id IN ({placeholders})", to_delete)
            conn.commit()
            deleted = cursor.rowcount
            cursor.close()
            conn.close()
            return jsonify({'message': f'{deleted} data berhasil dihapus', 'status': 'success'}), 200
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in delete_bulk_data (indices): {e}")
            if conn:
                conn.close()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Parameter "ids" atau "indices" diperlukan'}), 400


# ==================== TRAINING MODEL ====================
@bp.route('/api/train-model', methods=['POST', 'OPTIONS'])
@api_key_required
def train_model():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        # FIX #5: import lazy — hanya endpoint training yang butuh sklearn untuk fit().
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        start_time = time.time()

        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500

        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT pertanyaan, jawaban, kategori FROM dataset ORDER BY id")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'error': 'Dataset kosong'}), 400

        pertanyaan_list_train = [row['pertanyaan'] for row in rows]
        jawaban_list_train = [row['jawaban'] for row in rows]
        kategori_list_train = [row['kategori'] for row in rows]

        processed_questions = [preprocess(q) for q in pertanyaan_list_train]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_train_tfidf = vectorizer.fit_transform(processed_questions)

        y_train = kategori_list_train
        model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        model.fit(X_train_tfidf, y_train)

        qa_data_new = {
            'model': model,
            'vectorizer': vectorizer,
            'answers': jawaban_list_train,
            'questions': pertanyaan_list_train,
            'categories': kategori_list_train,
            # FIX #8: simpan matrix TF-IDF yang sudah dihitung supaya saat
            # model_service.load_models_and_data() dipanggil (tiap cold start), tidak perlu
            # preprocess()+transform() ulang semua pertanyaan dari nol.
            'X_all': X_train_tfidf,
        }

        supabase_saved = model_service.save_model_to_supabase(qa_data_new)
        if not supabase_saved:
            return jsonify({'error': 'Gagal menyimpan model ke Supabase'}), 500

        # Update memori global + invalidate cache
        model_service.model_qa = model
        model_service.vectorizer_qa = vectorizer
        model_service.answers = jawaban_list_train
        model_service.pertanyaan_list = pertanyaan_list_train
        model_service.kategori_list = kategori_list_train
        model_service._models_loaded = True
        # FIX #8: X_train_tfidf sudah dihitung di atas, pakai langsung — tidak perlu
        # _build_questions_cache() yang akan menjalankan preprocess()+transform() ulang.
        model_service._X_all_questions_cache = X_train_tfidf
        model_service._X_all_cache_version += 1

        return jsonify({
            'status': 'success',
            'message': 'Model Linear SVM berhasil dilatih berdasarkan kategori',
            'training_time': f'{time.time() - start_time:.2f} detik',
            'total_data': len(pertanyaan_list_train)
        }), 200

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return jsonify({'error': str(e)}), 500



@bp.route('/api/get-data/<int:index>', methods=['GET', 'OPTIONS'])
@api_key_required
def get_data_by_index(index):
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500
    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("SELECT id, pertanyaan, jawaban, kategori FROM dataset ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    if index < 0 or index >= len(rows):
        return jsonify({'error': 'Index tidak valid'}), 400
    row = rows[index]
    return jsonify({'index': index, 'id': row['id'], 'pertanyaan': row['pertanyaan'],
                    'jawaban': row['jawaban'], 'kategori': row['kategori']}), 200


@bp.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_dashboard_stats():
    if request.method == 'OPTIONS':
        return '', 200

    model_service.load_models_and_data()

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT COUNT(*) as total FROM admin")
        total_admin = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as total FROM pertanyaan_unknow")
        total_unknown = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as total FROM dataset")
        total_questions = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(DISTINCT kategori) as total FROM dataset")
        total_categories = cursor.fetchone()['total']
        cursor.close()
        conn.close()

        return jsonify({
            'total_admin': total_admin,
            'total_unknown_questions': total_unknown,
            'total_questions': total_questions,
            'total_categories': total_categories,
            'model_loaded': model_service.model_qa is not None
        }), 200
    except Exception as e:
        logger.error(f"Error in get_dashboard_stats: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/export-data', methods=['GET', 'OPTIONS'])
@api_key_required
def export_data():
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500
    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("SELECT pertanyaan, jawaban, kategori FROM dataset ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['pertanyaan', 'jawaban', 'kategori'])
    for row in rows:
        writer.writerow([row['pertanyaan'], row['jawaban'], row['kategori']])
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment;filename=dataset_export.csv'}), 200


@bp.route('/api/search-data', methods=['GET', 'OPTIONS'])
@api_key_required
def search_data():
    if request.method == 'OPTIONS':
        return '', 200
    query = request.args.get('q', '').strip()
    kategori = request.args.get('kategori', '').strip()
    if not query:
        return jsonify({'data': [], 'total': 0}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    cursor = get_db_cursor(conn, dictionary=True)
    base_sql = "SELECT id, pertanyaan, jawaban, kategori FROM dataset WHERE LOWER(pertanyaan) LIKE LOWER(%s)"
    params = [f"%{query}%"]
    if kategori:
        base_sql += " AND kategori = %s"
        params.append(kategori)
    base_sql += " ORDER BY id LIMIT 50"
    cursor.execute(base_sql, params)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    results = [{'index': row['id'] - 1, 'id': row['id'], 'pertanyaan': row['pertanyaan'],
                'jawaban': row['jawaban'], 'kategori': row['kategori']} for row in rows]
    return jsonify({'data': results, 'total': len(results)}), 200

