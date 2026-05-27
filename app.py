import sys
import traceback
import logging
import os
import csv
import time
import random
import threading
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv
import bcrypt
import jwt
import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
from preprocessing import preprocess
from supabase import create_client, Client  # type: ignore

# Load environment variables
load_dotenv()

# ===================== Konfigurasi Logging =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# ===================== Konfigurasi MySQL =====================
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))

FLASK_ENV = os.getenv("FLASK_ENV")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]

API_KEY = os.getenv("API_KEY")
API_KEY_HEADER = "X-API-Key"

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", 24))

# ===================== Inisialisasi Flask =====================
app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
     allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Requested-With"],
     supports_credentials=True,
     max_age=86400)

# ===================== KONFIGURASI SUPABASE =====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if SUPABASE_URL and SUPABASE_ANON_KEY else None
BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_FILE = os.getenv("MODEL_FILE")

# ===================== Path CSV =====================
csv_path = os.getenv("DATA_PATH")

# ==================== Connection Pool MySQL ====================
# FIX #1: Gunakan connection pool agar tidak buka koneksi baru tiap request
# Ini mengurangi latency 300-800ms per request ke Aiven
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
                pool_size=3,          # Hobby plan: jaga agar tidak overload Aiven free tier
                pool_reset_session=True,
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                port=MYSQL_PORT,
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
            conn = pool.get_connection()
            return conn
        except Error as e:
            logger.warning(f"Pool exhausted or error, trying direct connect: {e}")
    # Fallback ke direct connect
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            port=MYSQL_PORT,
            connection_timeout=10
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


# ==================== Global variables for model & data ====================
model_qa = None
vectorizer_qa = None
answers = []
pertanyaan_list = []
kategori_list = []
_models_loaded = False
_models_loading = False          # FIX #2: flag agar loading tidak dobel
_models_lock = threading.Lock()

# FIX #3: Cache matrix TF-IDF semua pertanyaan agar tidak dihitung ulang tiap request
_X_all_questions_cache = None
_X_all_cache_version = 0         # increment saat data berubah (setelah training)


# ===================== Handler OPTIONS =====================
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.status_code = 200
        return response

@app.route('/api/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return '', 200


# ===================== Helper Functions =====================
def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(admin_id, username):
    payload = {
        'admin_id': admin_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
    except:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token tidak ditemukan', 'authenticated': False}), 401
        if token.startswith('Bearer '):
            token = token[7:]
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Token tidak valid atau sudah kadaluarsa', 'authenticated': False}), 401
        request.admin = payload
        return f(*args, **kwargs)
    return decorated

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if FLASK_ENV == "development":
            return f(*args, **kwargs)
        api_key = request.headers.get(API_KEY_HEADER)
        if not api_key:
            return jsonify({'error': 'API Key tidak ditemukan', 'authenticated': False}), 401
        if api_key != API_KEY:
            return jsonify({'error': 'API Key tidak valid', 'authenticated': False}), 401
        return f(*args, **kwargs)
    return decorated

def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr


# ===================== Dataset CSV (fallback) =====================
def load_dataset_from_csv():
    """Load dataset dari CSV (fallback)."""
    q_list, a_list, k_list = [], [], []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    q_list.append(row[0].strip())
                    a_list.append(row[1].strip())
                    k_list.append(row[2].strip())
                elif len(row) == 2:
                    q_list.append(row[0].strip())
                    a_list.append(row[1].strip())
                    k_list.append('umum')
                elif len(row) == 1:
                    q_list.append(row[0].strip())
                    a_list.append('')
                    k_list.append('umum')
        logger.info(f"Loaded {len(q_list)} questions from CSV")
    except FileNotFoundError:
        logger.warning(f"CSV file not found at {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
    return q_list, a_list, k_list

def save_dataset_to_csv(q_list, a_list, k_list):
    """Simpan dataset ke CSV."""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['pertanyaan', 'jawaban', 'kategori'])
            for i in range(len(q_list)):
                writer.writerow([
                    q_list[i],
                    a_list[i] if i < len(a_list) else '',
                    k_list[i] if i < len(k_list) else 'umum'
                ])
        logger.info(f"Saved {len(q_list)} questions to CSV")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")


# ===================== Load Dataset & Model =====================
def load_dataset_from_db():
    """Load dataset dari tabel 'dataset' ke global variables."""
    global pertanyaan_list, answers, kategori_list
    conn = get_db_connection()
    if conn is None:
        logger.error("Database connection failed, cannot load dataset")
        return
    try:
        cursor = get_db_cursor(conn, dictionary=True)
        cursor.execute("SELECT pertanyaan, jawaban, kategori FROM dataset ORDER BY id")
        rows = cursor.fetchall()
        pertanyaan_list = [row['pertanyaan'] for row in rows]
        answers = [row['jawaban'] for row in rows]
        kategori_list = [row['kategori'] for row in rows]
        logger.info(f"Dataset loaded from database: {len(pertanyaan_list)} questions")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error loading dataset from database: {e}")
        if conn:
            conn.close()

def load_model_from_supabase():
    """Load model dari Supabase Storage ke memori."""
    if supabase is None:
        logger.warning("Supabase not configured")
        return None
    try:
        t0 = time.time()
        file_data = supabase.storage.from_(BUCKET_NAME).download(MODEL_FILE)
        model_data = pickle.loads(file_data)
        logger.info(f"Model loaded from Supabase in {time.time()-t0:.2f}s")
        return model_data
    except Exception as e:
        logger.error(f"Failed to load model from Supabase: {e}")
        return None

def save_model_to_supabase(model_data):
    """Simpan model ke Supabase Storage."""
    if supabase is None:
        logger.warning("Supabase not configured, cannot save model")
        return False
    try:
        model_bytes = pickle.dumps(model_data)
        try:
            supabase.storage.from_(BUCKET_NAME).upload(
                MODEL_FILE, model_bytes,
                file_options={"content-type": "application/octet-stream"}
            )
            logger.info("Model saved to Supabase storage (new file)")
        except Exception:
            supabase.storage.from_(BUCKET_NAME).update(
                MODEL_FILE, model_bytes,
                file_options={"content-type": "application/octet-stream"}
            )
            logger.info("Model updated in Supabase storage")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to Supabase: {e}")
        return False

def _build_questions_cache():
    """
    FIX #3: Pre-compute dan cache matrix TF-IDF semua pertanyaan.
    Dipanggil sekali setelah model pertama kali dimuat atau setelah training.
    """
    global _X_all_questions_cache
    if vectorizer_qa is None or not pertanyaan_list:
        return
    try:
        t0 = time.time()
        _X_all_questions_cache = vectorizer_qa.transform(
            [preprocess(q) for q in pertanyaan_list]
        )
        logger.info(f"TF-IDF matrix pre-computed ({len(pertanyaan_list)} rows) in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"Error building questions cache: {e}")
        _X_all_questions_cache = None

def load_models_and_data():
    """
    FIX #2: Thread-safe lazy loading dengan flag _models_loading agar
    dua request bersamaan tidak men-download model dua kali dari Supabase.
    """
    global model_qa, vectorizer_qa, answers, pertanyaan_list, kategori_list
    global _models_loaded, _models_loading, _X_all_questions_cache

    if _models_loaded:
        return

    with _models_lock:
        if _models_loaded:
            return
        if _models_loading:
            # Request lain sedang loading, tunggu sampai selesai
            logger.info("Model sedang dimuat oleh thread lain, menunggu...")
            while _models_loading and not _models_loaded:
                time.sleep(0.1)
            return

        _models_loading = True

    try:
        model_data = load_model_from_supabase()
        if model_data:
            model_qa = model_data['model']
            vectorizer_qa = model_data['vectorizer']
            answers = model_data['answers']
            pertanyaan_list = model_data['questions']
            kategori_list = model_data.get('categories', [])
            logger.info(f"Model loaded from Supabase: {len(pertanyaan_list)} questions")
            _build_questions_cache()   # FIX #3: build cache langsung setelah load
        else:
            load_dataset_from_db()
            logger.warning("No model available, please train first")
    finally:
        with _models_lock:
            _models_loaded = True
            _models_loading = False


# ===================== Background Warmup =====================
# FIX #4: Muat model di background saat aplikasi pertama kali start.
# Dengan ini, saat user pertama kirim chat, model sudah siap di memori.
def _background_warmup():
    """Jalankan warmup di background thread agar tidak memblokir startup."""
    logger.info("[Warmup] Background model loading started...")
    try:
        load_models_and_data()
        # Pemanasan koneksi database
        conn = get_db_connection()
        if conn:
            conn.close()
            logger.info("[Warmup] DB connection warmed up")
        logger.info("[Warmup] Complete. Model ready.")
    except Exception as e:
        logger.error(f"[Warmup] Error: {e}")

# Jalankan warmup saat modul dimuat (gunicorn worker start)
_warmup_thread = threading.Thread(target=_background_warmup, daemon=True)
_warmup_thread.start()


# ===================== Profanity Filter =====================
BAD_WORDS = list(set([
    # Indonesia umum
    "anjing", "babi", "kontol", "memek", "jembut", "tai", "goblok", "bodoh",
    "tolol", "idiot", "sialan", "brengsek", "bangsat", "kampret", "kampang",
    "keparat", "ngentot", "ngewe", "setan", "iblis", "ngaco",
    # Jawa
    "asu", "asw", "jancuk", "jancok", "diancuk", "cuk", "ndasmu", "matamu", "bajingan",
    "tempek", "tempik", "perek", "gendeng", "edan", "gemblung", "dongo", "kebo", "kodok",
    # Sunda
    "bangsad", "belog", "torog", "kohok", "teu", "aing", "dare", "bebek", "haseum",
    # Batak
    "bodat", "begu", "haporason", "jabud", "baito", "lappang", "sihit", "sipinggan",
    # Minang
    "indak", "karuah", "kawa", "taku", "cuak", "bajang",
    # Bugis/Makassar
    "pate", "lokka", "bangkeng", "curang",
    # Maluku
    "pukul", "fufu", "sale", "puki", "kolot", "sogo", "bampuki", "pepek",
    "kimai", "cuki", "kudacuki", "kuda cuki", "cukimai", "bampukar",
    # Papua
    "bangke", "kuskus",
    # Inggris
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy", "cunt",
    "whore", "slut", "motherfucker", "damn", "hell", "stupid", "moron",
    # Variasi typo
    "kont*l", "kont0l", "k0nt0l", "m3m3k", "mem3k", "anj1ng", "4nj1ng", "b4b1", "b4bi",
    "gobl0k", "b0d0h", "ancuk", "janc0k"
]))

PROFANITY_RESPONSES = [
    "Maaf, pertanyaan Anda mengandung kata-kata yang tidak pantas. Harap ajukan pertanyaan dengan sopan. Penggunaan kata kasar tidak akan dilayani dan tidak akan disimpan.",
    "Kami tidak dapat memproses pertanyaan yang mengandung kata kasar. Silakan ajukan pertanyaan dengan bahasa yang baik dan benar. Terima kasih.",
    "Mohon untuk tidak menggunakan kata-kata kasar. Chatbot ini dirancang untuk membantu dengan ramah. Ulangi pertanyaan Anda dengan sopan.",
    "Pertanyaan Anda mengandung kata tidak pantas. Sebagai bentuk edukasi, hindari kata kasar agar percakapan tetap nyaman. Silakan coba lagi.",
    "Kata kasar terdeteksi. Kami tidak akan menyimpan pertanyaan ini. Harap gunakan bahasa yang santun."
]

def get_profanity_response():
    return random.choice(PROFANITY_RESPONSES)

def contains_profanity(text):
    if not text:
        return False
    text_lower = text.lower()
    for word in BAD_WORDS:
        if word in text_lower:
            return True
    return False


# ===================== Utility =====================
def save_unknown_question(question):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO pertanyaan_unknow (pertanyaan) VALUES (%s)", (question,))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Save unknown error: {e}")
    else:
        logger.info(f"Unknown question (not saved): {question}")


# ==================== ENDPOINTS =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_qa is not None,
        'model_loading': _models_loading,
        'dataset_size': len(pertanyaan_list)
    })

# FIX #4 (lanjutan): Endpoint warmup yang bisa dipanggil oleh cron/uptime monitor
# agar Vercel instance tidak cold. Panggil endpoint ini setiap 4 menit dari
# layanan cron gratis seperti cron-job.org atau UptimeRobot.
@app.route('/api/warmup', methods=['GET'])
def warmup():
    """
    Endpoint ringan untuk mencegah cold start.
    Daftarkan URL ini ke UptimeRobot / cron-job.org dengan interval 4 menit.
    """
    already_loaded = _models_loaded
    if not already_loaded:
        # Jika belum loaded, trigger di background (non-blocking)
        if not _models_loading:
            t = threading.Thread(target=load_models_and_data, daemon=True)
            t.start()
    return jsonify({
        'status': 'ok',
        'model_ready': _models_loaded,
        'dataset_size': len(pertanyaan_list),
        'timestamp': datetime.now().isoformat()
    })


# ==================== CHAT ENDPOINTS ====================
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
@api_key_required
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    user_input = request.json.get('pertanyaan') or request.json.get('question') or ''
    if not user_input:
        return jsonify({'error': 'Pertanyaan kosong'}), 400

    # 1. Proteksi Kata Kasar
    if contains_profanity(user_input):
        return jsonify({
            'pertanyaan': user_input,
            'jawaban': get_profanity_response(),
            'status': 'error'
        }), 200

    # 2. Pastikan model sudah tersedia (sudah di-warmup di background)
    load_models_and_data()
    if model_qa is None or vectorizer_qa is None:
        return jsonify({'error': 'Model belum siap, lakukan training terlebih dahulu.'}), 500

    # 3. Preprocessing & Transform Input User ke TF-IDF
    processed_input = preprocess(user_input)
    X_input_tfidf = vectorizer_qa.transform([processed_input])

    if X_input_tfidf.nnz == 0:
        save_unknown_question(user_input)
        return jsonify({
            'pertanyaan': user_input,
            'jawaban': "Mohon maaf, saya belum mengerti pertanyaan Anda.",
            'status': 'unknown'
        }), 200

    # 4. Prediksi Kategori dengan Linear SVM
    predicted_category = model_qa.predict(X_input_tfidf)[0]

    # 5. FIX #3: Gunakan cache matrix TF-IDF (tidak re-compute setiap request)
    global _X_all_questions_cache
    if _X_all_questions_cache is None:
        _build_questions_cache()
    if _X_all_questions_cache is None:
        save_unknown_question(user_input)
        return jsonify({
            'pertanyaan': user_input,
            'jawaban': "Mohon maaf, saya belum mengerti pertanyaan Anda.",
            'status': 'unknown'
        }), 200

    predict_answer_score = (X_input_tfidf * _X_all_questions_cache.T).toarray().flatten()

    # 6. Filter per kategori
    category_indices = [idx for idx, cat in enumerate(kategori_list) if cat == predicted_category]

    if not category_indices:
        save_unknown_question(user_input)
        return jsonify({
            'pertanyaan': user_input,
            'jawaban': "Mohon maaf, saya belum mengerti pertanyaan Anda.",
            'status': 'unknown'
        }), 200

    best_index = max(category_indices, key=lambda idx: predict_answer_score[idx])
    max_predict_score = predict_answer_score[best_index]

    if max_predict_score < 0.15:
        save_unknown_question(user_input)
        return jsonify({
            'pertanyaan': user_input,
            'jawaban': "Mohon maaf, saya belum mengerti pertanyaan Anda.",
            'status': 'unknown'
        }), 200

    return jsonify({
        'pertanyaan': user_input,
        'jawaban': str(answers[best_index]),
        'status': 'ok',
        'kategori': str(predicted_category),
        'confidence_score': float(max_predict_score)
    }), 200


@app.route('/api/chat-n8n-proxy', methods=['POST', 'OPTIONS'])
@api_key_required
def chat_n8n_proxy():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json
    if not data or 'pertanyaan' not in data:
        return jsonify({'error': 'Pertanyaan tidak boleh kosong'}), 400

    user_input = data['pertanyaan']
    logger.info(f"[Optimasi Token] Memproses Hybrid AI untuk: {user_input}")

    jawaban_dasar_svm = "Mohon maaf, saya belum mengerti pertanyaan Anda."
    kategori_terdeteksi = "unknown"
    referensi_alternatif_text = "TIDAK_ADA_ALTERNATIF"

    if contains_profanity(user_input):
        jawaban_dasar_svm = get_profanity_response()
        kategori_terdeteksi = "profanity"
    else:
        load_models_and_data()
        if model_qa is None or vectorizer_qa is None:
            return jsonify({'error': 'Model belum siap, lakukan training terlebih dahulu.'}), 500

        processed_input = preprocess(user_input)
        X_input_tfidf = vectorizer_qa.transform([processed_input])

        if X_input_tfidf.nnz > 0:
            predicted_category = model_qa.predict(X_input_tfidf)[0]

            # FIX #3: Gunakan cache
            global _X_all_questions_cache
            if _X_all_questions_cache is None:
                _build_questions_cache()

            if _X_all_questions_cache is not None:
                predict_answer_score = (X_input_tfidf * _X_all_questions_cache.T).toarray().flatten()
                category_indices = [idx for idx, cat in enumerate(kategori_list) if cat == predicted_category]

                if category_indices:
                    best_index = max(category_indices, key=lambda idx: predict_answer_score[idx])
                    max_predict_score = predict_answer_score[best_index]

                    if max_predict_score >= 0.35:
                        jawaban_dasar_svm = str(answers[best_index])
                        kategori_terdeteksi = str(predicted_category)
                    else:
                        save_unknown_question(user_input)

                    top_global_indices = sorted(
                        range(len(predict_answer_score)),
                        key=lambda i: predict_answer_score[i],
                        reverse=True
                    )[:3]

                    lines = []
                    for rank, idx in enumerate(top_global_indices):
                        if predict_answer_score[idx] > 0.05:
                            lines.append(
                                f"Alternatif {rank+1}:\nPertanyaan: {pertanyaan_list[idx]}\nJawaban: {answers[idx]}"
                            )
                    if lines:
                        referensi_alternatif_text = "\n\n".join(lines)
                else:
                    save_unknown_question(user_input)
        else:
            save_unknown_question(user_input)

    n8n_webhook_url = "https://pasastimuslim.app.n8n.cloud/webhook/royal-resto-qa"
    headers_n8n = {
        "Content-Type": "application/json",
        "X-API-Key": "hG&*g^td&^@!%*^98*$%hY12^%75*!@*%uiy*^&^rs75&&^^FTF*%"
    }
    payload_n8n = {
        "pertanyaan": str(user_input),
        "jawaban_svm": str(jawaban_dasar_svm),
        "kategori": str(kategori_terdeteksi),
        "referensi_alternatif": str(referensi_alternatif_text)
    }

    try:
        import requests
        response_n8n = requests.post(n8n_webhook_url, json=payload_n8n, headers=headers_n8n, timeout=25)
        if response_n8n.status_code == 200:
            return jsonify(response_n8n.json()), 200
        else:
            return jsonify({'status': 'success', 'answer': jawaban_dasar_svm}), 200
    except Exception as e:
        logger.error(f"Koneksi n8n RTO/Gagal, mengaktifkan fallback: {str(e)}")
        return jsonify({'status': 'success', 'answer': jawaban_dasar_svm}), 200


@app.route('/api/chat/ambiguous-unknown', methods=['POST', 'OPTIONS'])
@api_key_required
def handle_ambiguous_unknown():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    original_question = data.get('original_question', '').strip()

    if not original_question:
        return jsonify({'error': 'Pertanyaan asli tidak ditemukan'}), 400

    if contains_profanity(original_question):
        return jsonify({
            'status': 'error',
            'jawaban': get_profanity_response(),
            'original_question': original_question
        }), 200

    save_unknown_question(original_question)

    return jsonify({
        'status': 'unknown',
        'jawaban': 'Mohon maaf, saya belum mengerti pertanyaan Anda. Pertanyaan Anda telah saya catat untuk pembelajaran ke depannya.',
        'original_question': original_question
    }), 200


# ==================== RECOMMENDATIONS ====================
@app.route('/api/recommendations/random', methods=['GET', 'OPTIONS'])
@api_key_required
def get_random_recommendations():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        cursor = conn.cursor()
        cursor.execute("SELECT pertanyaan FROM dataset ORDER BY RAND() LIMIT 3")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        recommendations = [row[0] for row in rows]
        return jsonify({'recommendations': recommendations}), 200
    except Exception as e:
        logger.error(f"Error in get_random_recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/by-category', methods=['GET', 'OPTIONS'])
@api_key_required
def get_recommendations_by_category():
    if request.method == 'OPTIONS':
        return '', 200
    kategori = request.args.get('kategori')
    if not kategori:
        return jsonify({'error': 'Parameter kategori diperlukan'}), 400
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database tidak tersedia'}), 500
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pertanyaan FROM dataset WHERE kategori = %s ORDER BY RAND() LIMIT 3",
            (kategori,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        recommendations = [row[0] for row in rows]
        return jsonify({'recommendations': recommendations}), 200
    except Exception as e:
        logger.error(f"Error in recommendations by category: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== AUTHENTICATION ====================
@app.route('/api/login', methods=['POST', 'OPTIONS'])
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


@app.route('/api/logout', methods=['POST', 'OPTIONS'])
@api_key_required
@token_required
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'message': 'Logout berhasil', 'authenticated': False}), 200


@app.route('/api/verify-token', methods=['GET', 'OPTIONS'])
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


@app.route('/api/change-password', methods=['POST', 'OPTIONS'])
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


@app.route('/api/verify-admin-password', methods=['POST', 'OPTIONS'])
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


@app.route('/api/admin-profile', methods=['GET', 'OPTIONS'])
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
@app.route('/api/admins', methods=['GET', 'OPTIONS'])
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


@app.route('/api/add/admin', methods=['POST', 'OPTIONS'])
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


@app.route('/api/admins/<int:admin_id>', methods=['PUT', 'OPTIONS'])
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


@app.route('/api/admins/<int:admin_id>/reset-password', methods=['POST', 'OPTIONS'])
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


@app.route('/api/admins/<int:admin_id>', methods=['DELETE', 'OPTIONS'])
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


# ==================== UNKNOWN QUESTIONS ====================
@app.route('/api/pertanyaan-unknown', methods=['GET', 'OPTIONS'])
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


@app.route('/api/delete-unknown', methods=['DELETE', 'OPTIONS'])
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


@app.route('/api/delete-all-unknown', methods=['DELETE', 'OPTIONS'])
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
@app.route('/api/kategori', methods=['GET', 'OPTIONS'])
@api_key_required
def get_kategori():
    if request.method == 'OPTIONS':
        return '', 200
    load_models_and_data()
    categories = sorted(list(set(kategori_list)))
    return jsonify({'kategori': categories})


@app.route('/api/model-info', methods=['GET', 'OPTIONS'])
@api_key_required
def model_info():
    if request.method == 'OPTIONS':
        return '', 200
    load_models_and_data()
    return jsonify({
        'total_questions': len(pertanyaan_list),
        'total_answers': len(answers),
        'categories': sorted(list(set(kategori_list))),
        'model_loaded': model_qa is not None,
        'vectorizer_loaded': vectorizer_qa is not None
    }), 200


# ==================== DATASET MANAGEMENT ====================
@app.route('/api/get-all-data', methods=['GET', 'OPTIONS'])
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


@app.route('/api/unknown/recent', methods=['GET', 'OPTIONS'])
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


@app.route('/api/dataset/recent', methods=['GET', 'OPTIONS'])
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


@app.route('/api/tambah-data', methods=['POST', 'OPTIONS'])
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


@app.route('/api/update-data', methods=['PUT', 'OPTIONS'])
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


@app.route('/api/delete-data', methods=['DELETE', 'OPTIONS'])
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


@app.route('/api/delete-bulk-data', methods=['DELETE', 'OPTIONS'])
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
@app.route('/api/train-model', methods=['POST', 'OPTIONS'])
@api_key_required
def train_model():
    if request.method == 'OPTIONS':
        return '', 200
    try:
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
            'categories': kategori_list_train
        }

        supabase_saved = save_model_to_supabase(qa_data_new)
        if not supabase_saved:
            return jsonify({'error': 'Gagal menyimpan model ke Supabase'}), 500

        # Update memori global + invalidate cache
        global model_qa, vectorizer_qa, answers, pertanyaan_list, kategori_list
        global _models_loaded, _X_all_questions_cache, _X_all_cache_version
        model_qa = model
        vectorizer_qa = vectorizer
        answers = jawaban_list_train
        pertanyaan_list = pertanyaan_list_train
        kategori_list = kategori_list_train
        _models_loaded = True
        _X_all_questions_cache = None   # Invalidate cache
        _X_all_cache_version += 1
        _build_questions_cache()        # Rebuild cache langsung setelah training

        return jsonify({
            'status': 'success',
            'message': 'Model Linear SVM berhasil dilatih berdasarkan kategori',
            'training_time': f'{time.time() - start_time:.2f} detik',
            'total_data': len(pertanyaan_list_train)
        }), 200

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== DEBUG ====================
@app.route('/api/cek-csv', methods=['GET', 'OPTIONS'])
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


@app.route('/api/fix-csv', methods=['POST', 'OPTIONS'])
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
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['pertanyaan', 'jawaban', 'kategori'])
            for row in rows:
                writer.writerow([row['pertanyaan'], row['jawaban'], row['kategori']])
        return jsonify({'message': 'CSV berhasil diperbaiki dari database', 'total_rows': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug-db')
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


@app.route('/api/test-db')
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


@app.route('/api/debug-headers', methods=['GET'])
def debug_headers():
    return jsonify({
        "message": "Cek logs untuk detail",
        "origin": request.headers.get('Origin', 'Tidak ada origin'),
        "user_agent": request.headers.get('User-Agent', 'Tidak ada user-agent'),
        "host": request.headers.get('Host', 'Tidak ada host'),
        "x-forwarded-for": request.headers.get('X-Forwarded-For', 'Tidak ada info IP'),
    }), 200


# ==================== ADDITIONAL ENDPOINTS ====================
@app.route('/api/get-data/<int:index>', methods=['GET', 'OPTIONS'])
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


@app.route('/api/register', methods=['POST', 'OPTIONS'])
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


@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_dashboard_stats():
    if request.method == 'OPTIONS':
        return '', 200

    load_models_and_data()

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
            'model_loaded': model_qa is not None
        }), 200
    except Exception as e:
        logger.error(f"Error in get_dashboard_stats: {e}")
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-data', methods=['GET', 'OPTIONS'])
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


@app.route('/api/search-data', methods=['GET', 'OPTIONS'])
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


@app.route('/api/login-logs', methods=['GET', 'OPTIONS'])
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


@app.route('/api/reset-database', methods=['POST', 'OPTIONS'])
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


# ==================== Global Error Handler ====================
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'Internal server error', 'message': str(e) if FLASK_DEBUG else 'Terjadi kesalahan'}), 500


if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host='0.0.0.0', port=FLASK_PORT)