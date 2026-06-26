import logging

from flask import Blueprint, jsonify, request

from api import config
from api.auth import (
    api_key_required,
    generate_token,
    get_client_ip,
    hash_password,
    token_required,
    verify_password,
    verify_token,
)
from api.database import get_db_connection, get_db_cursor

logger = logging.getLogger(__name__)
bp = Blueprint('auth_admin', __name__)

JWT_EXPIRATION_HOURS = config.JWT_EXPIRATION_HOURS


@bp.route('/api/login', methods=['POST', 'OPTIONS'])
@api_key_required
def login():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': 'Username dan password harus diisi', 'authenticated': False}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia', 'authenticated': False}), 500

    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute(
            "SELECT id, username, password, email, full_name, is_active FROM admin WHERE username = %s",
            (username,)
        )
        admin = cursor.fetchone()
        if not admin:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Username atau password salah', 'authenticated': False}), 401

        if not admin['is_active']:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Akun Anda telah dinonaktifkan.', 'authenticated': False}), 401

        if not verify_password(password, admin['password']):
            cursor.close()
            conn.close()
            return jsonify({'error': 'Username atau password salah', 'authenticated': False}), 401

        update_cursor = conn.cursor()
        update_cursor.execute("UPDATE admin SET last_login = NOW() WHERE id = %s", (admin['id'],))
        conn.commit()
        update_cursor.close()

        token = generate_token(admin['id'], admin['username'])

        # Log login
        try:
            log_cursor = conn.cursor()
            log_cursor.execute(
                "INSERT INTO login_logs (admin_id, ip_address, login_time) VALUES (%s, %s, NOW())",
                (admin['id'], get_client_ip())
            )
            conn.commit()
            log_cursor.close()
        except Exception:
            pass  # Login log bukan blocker

        cursor.close()
        conn.close()

        return jsonify({
            'authenticated': True,
            'token': token,
            'admin': {
                'id': admin['id'],
                'username': admin['username'],
                'email': admin['email'],
                'full_name': admin['full_name']
            },
            'expires_in': JWT_EXPIRATION_HOURS * 3600
        }), 200

    except Exception as e:
        logger.error(f"Error in login: {e}")
        if conn:
            conn.close()
        return jsonify({'error': 'Terjadi kesalahan saat login', 'authenticated': False}), 500


@bp.route('/api/logout', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'message': 'Logout berhasil', 'authenticated': False}), 200


@bp.route('/api/verify-token', methods=['GET', 'OPTIONS'])
@api_key_required
def verify_token_endpoint():
    if request.method == 'OPTIONS':
        return '', 200
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'authenticated': False}), 401
    if token.startswith('Bearer '):
        token = token[7:]
    payload = verify_token(token)
    if not payload:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'admin_id': payload['admin_id'],
        'username': payload['username']
    }), 200


@bp.route('/api/change-password', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def change_password():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}
    old_password = data.get('old_password', '').strip()
    new_password = data.get('new_password', '').strip()
    confirm_password = data.get('confirm_password', '').strip()
    if not old_password or not new_password or not confirm_password:
        return jsonify({'error': 'Semua field harus diisi'}), 400
    if new_password != confirm_password:
        return jsonify({'error': 'Password baru tidak cocok'}), 400
    if len(new_password) < 6:
        return jsonify({'error': 'Password minimal 6 karakter'}), 400

    admin_id = request.admin['admin_id']
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Password berhasil diubah (demo)'}), 200

    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("SELECT password FROM admin WHERE id = %s", (admin_id,))
    admin = cursor.fetchone()
    if not admin:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Admin tidak ditemukan'}), 404
    if not verify_password(old_password, admin['password']):
        cursor.close()
        conn.close()
        return jsonify({'error': 'Password lama salah'}), 401

    new_hash = hash_password(new_password)
    cursor = conn.cursor()
    cursor.execute("UPDATE admin SET password = %s, updated_at = NOW() WHERE id = %s", (new_hash, admin_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Password berhasil diubah'}), 200


@bp.route('/api/verify-admin-password', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def verify_admin_password():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    password = data.get('password', '').strip()
    if not password:
        return jsonify({'error': 'Password harus diisi', 'valid': False}), 400

    admin_id = request.admin['admin_id']
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia', 'valid': False}), 500

    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT password FROM admin WHERE id = %s", (admin_id,))
        admin = cursor.fetchone()
        cursor.close()
        conn.close()

        if not admin:
            return jsonify({'error': 'Admin tidak ditemukan', 'valid': False}), 404

        if verify_password(password, admin['password']):
            return jsonify({'valid': True, 'message': 'Password valid'}), 200
        else:
            return jsonify({'valid': False, 'error': 'Password salah'}), 401

    except Exception as e:
        logger.error(f"Error in verify_admin_password: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e), 'valid': False}), 500


@bp.route('/api/admin-profile', methods=['GET', 'OPTIONS'])
@api_key_required
@token_required
def get_admin_profile():
    if request.method == 'OPTIONS':
        return '', 200
    admin_id = request.admin['admin_id']
    conn = get_db_connection()
    if conn is None:
        return jsonify({'admin': {'id': admin_id, 'username': 'admin'}}), 200
    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute(
        "SELECT id, username, email, full_name, is_active, last_login, created_at FROM admin WHERE id = %s",
        (admin_id,)
    )
    admin = cursor.fetchone()
    cursor.close()
    conn.close()
    if admin:
        if admin.get('last_login'):
            admin['last_login'] = admin['last_login'].strftime('%Y-%m-%d %H:%M:%S')
        if admin.get('created_at'):
            admin['created_at'] = admin['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({'admin': admin}), 200


# ==================== KELOLA ADMIN ====================
@bp.route('/api/admins', methods=['GET', 'OPTIONS'])
@api_key_required
@token_required
def get_all_admins():
    if request.method == 'OPTIONS':
        return '', 200

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '', type=str)
    offset = (page - 1) * per_page

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = get_db_cursor(conn, dictionary=True)
        query = "SELECT id, username, email, full_name, is_active, last_login, created_at FROM admin"
        params = []
        if search:
            query += " WHERE (LOWER(username) LIKE LOWER(%s) OR LOWER(email) LIKE LOWER(%s) OR LOWER(full_name) LIKE LOWER(%s))"
            sp = f"%{search}%"
            params.extend([sp, sp, sp])
        query += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        admins = cursor.fetchall()

        count_query = "SELECT COUNT(*) as total FROM admin"
        if search:
            count_query += " WHERE (LOWER(username) LIKE LOWER(%s) OR LOWER(email) LIKE LOWER(%s) OR LOWER(full_name) LIKE LOWER(%s))"
            cursor.execute(count_query, [sp, sp, sp])
        else:
            cursor.execute(count_query)
        total = cursor.fetchone()['total']
        total_pages = (total + per_page - 1) // per_page

        for admin in admins:
            if admin.get('last_login'):
                admin['last_login'] = admin['last_login'].strftime('%Y-%m-%d %H:%M:%S')
            if admin.get('created_at'):
                admin['created_at'] = admin['created_at'].strftime('%Y-%m-%d %H:%M:%S')

        cursor.close()
        conn.close()
        return jsonify({'page': page, 'per_page': per_page, 'total_data': total, 'total_pages': total_pages, 'data': admins}), 200
    except Exception as e:
        logger.error(f"Error in get_all_admins: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/add/admin', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def create_admin():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    full_name = data.get('full_name', '').strip()
    is_active = data.get('is_active', True)

    if not username or not password or not email or not full_name:
        return jsonify({'error': 'Semua field harus diisi'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password minimal 6 karakter'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM admin WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Username sudah digunakan'}), 409
        cursor.execute("SELECT id FROM admin WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email sudah digunakan'}), 409

        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO admin (username, password, email, full_name, is_active) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed, email, full_name, int(is_active))
        )
        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return jsonify({'message': 'Admin berhasil dibuat', 'admin_id': new_id}), 201
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating admin: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/admins/<int:admin_id>', methods=['PUT', 'OPTIONS'])
@api_key_required
@token_required
def update_admin(admin_id):
    if request.method == 'OPTIONS':
        return '', 200

    current_admin_id = request.admin['admin_id']
    if current_admin_id != admin_id:
        return jsonify({'error': 'Anda hanya dapat mengedit akun Anda sendiri'}), 403

    data = request.json or {}
    email = data.get('email', '').strip()
    full_name = data.get('full_name', '').strip()
    is_active = data.get('is_active', True)

    if not email or not full_name:
        return jsonify({'error': 'Email dan Nama Lengkap harus diisi'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM admin WHERE id = %s", (admin_id,))
        if not cursor.fetchone():
            return jsonify({'error': 'Admin tidak ditemukan'}), 404
        cursor.execute("SELECT id FROM admin WHERE email = %s AND id != %s", (email, admin_id))
        if cursor.fetchone():
            return jsonify({'error': 'Email sudah digunakan oleh admin lain'}), 409
        cursor.execute(
            "UPDATE admin SET email=%s, full_name=%s, is_active=%s, updated_at=NOW() WHERE id=%s",
            (email, full_name, int(is_active), admin_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Admin berhasil diupdate'}), 200
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating admin: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/admins/<int:admin_id>/reset-password', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def reset_admin_password(admin_id):
    if request.method == 'OPTIONS':
        return '', 200

    current_admin_id = request.admin['admin_id']
    if current_admin_id != admin_id:
        return jsonify({'error': 'Anda hanya dapat mereset password akun Anda sendiri'}), 403

    data = request.json or {}
    new_password = data.get('new_password', '').strip()
    if not new_password or len(new_password) < 6:
        return jsonify({'error': 'Password minimal 6 karakter'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM admin WHERE id = %s", (admin_id,))
        if not cursor.fetchone():
            return jsonify({'error': 'Admin tidak ditemukan'}), 404
        hashed = hash_password(new_password)
        cursor.execute("UPDATE admin SET password=%s, updated_at=NOW() WHERE id=%s", (hashed, admin_id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Password berhasil direset'}), 200
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resetting password: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/admins/<int:admin_id>', methods=['DELETE', 'OPTIONS'])
@api_key_required
@token_required
def delete_admin(admin_id):
    if request.method == 'OPTIONS':
        return '', 200

    current_admin_id = request.admin['admin_id']
    if current_admin_id == admin_id:
        return jsonify({'error': 'Anda tidak dapat menghapus akun Anda sendiri'}), 403

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database tidak tersedia'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM admin WHERE id = %s", (admin_id,))
        if not cursor.fetchone():
            return jsonify({'error': 'Admin tidak ditemukan'}), 404
        cursor.execute("DELETE FROM admin WHERE id = %s", (admin_id,))
        conn.commit()
        logger.info(f"Admin {current_admin_id} menghapus admin {admin_id}")
        cursor.close()
        conn.close()
        return jsonify({'message': 'Admin berhasil dihapus', 'deleted_id': admin_id}), 200
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting admin: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500



@bp.route('/api/register', methods=['POST', 'OPTIONS'])
@api_key_required
def register_admin():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    full_name = data.get('full_name', '').strip()
    if not username or not password or not email or not full_name:
        return jsonify({'error': 'Semua field harus diisi'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password minimal 6 karakter'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Registrasi berhasil (demo)', 'admin_id': 999}), 201

    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("SELECT id FROM admin WHERE username = %s", (username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({'error': 'Username sudah digunakan'}), 409
    cursor.execute("SELECT id FROM admin WHERE email = %s", (email,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({'error': 'Email sudah digunakan'}), 409
    hashed = hash_password(password)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO admin (username, password, email, full_name, is_active) VALUES (%s, %s, %s, %s, 1)",
                   (username, hashed, email, full_name))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({'message': 'Registrasi berhasil', 'admin_id': new_id}), 201



@bp.route('/api/login-logs', methods=['GET', 'OPTIONS'])
@api_key_required
@token_required
def get_login_logs():
    if request.method == 'OPTIONS':
        return '', 200
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    offset = (page - 1) * per_page

    conn = get_db_connection()
    if conn is None:
        return jsonify({'data': [], 'total_data': 0}), 200
    cursor = get_db_cursor(conn, dictionary=True)
    cursor.execute("""
        SELECT l.*, a.username as admin_username
        FROM login_logs l
        LEFT JOIN admin a ON l.admin_id = a.id
        ORDER BY l.login_time DESC
        LIMIT %s OFFSET %s
    """, (per_page, offset))
    logs = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) as total FROM login_logs")
    total = cursor.fetchone()['total']
    total_pages = (total + per_page - 1) // per_page
    cursor.close()
    conn.close()
    for log in logs:
        if log.get('login_time'):
            log['login_time'] = log['login_time'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({'page': page, 'per_page': per_page, 'total_data': total,
                    'total_pages': total_pages, 'data': logs}), 200


@bp.route('/api/reset-database', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def reset_database():
    if request.method == 'OPTIONS':
        return '', 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Database berhasil direset (demo)', 'deleted_unknown_questions': 0}), 200
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pertanyaan_unknow")
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Database berhasil direset', 'deleted_unknown_questions': deleted}), 200


