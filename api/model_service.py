import csv
import gzip
import logging
import os
import pickle
import random
import time
import threading

from preprocessing import preprocess

from api import config
from api.database import get_db_connection, get_db_cursor

logger = logging.getLogger(__name__)
DATA_PATH = config.DATA_PATH



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

def load_dataset_from_csv():
    """Load dataset dari CSV (fallback)."""
    q_list, a_list, k_list = [], [], []
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
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
        logger.warning(f"CSV file not found at {DATA_PATH}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
    return q_list, a_list, k_list

def save_dataset_to_csv(q_list, a_list, k_list):
    """Simpan dataset ke CSV."""
    try:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, 'w', encoding='utf-8', newline='') as f:
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
    """Load model dari Supabase Storage ke memori.

    FIX #6: Timing per-tahap (download / decompress / unpickle) agar mudah
    melihat di log tahap mana yang sebenarnya paling lambat.
    FIX #7: Mendukung file model yang sudah dikompresi gzip (lihat
    save_model_to_supabase). Tetap backward-compatible dengan file lama
    yang belum dikompresi — jika gzip.decompress gagal, dianggap pickle
    mentah (perilaku lama).
    """
    if config.supabase is None:
        logger.warning("Supabase not configured")
        return None
    if not config.BUCKET_NAME or not config.MODEL_FILE:
        logger.warning("BUCKET_NAME atau MODEL_FILE tidak dikonfigurasi")
        return None
    try:
        t0 = time.time()
        logger.info(f"[Supabase] Mulai download model: bucket={config.BUCKET_NAME}, file={config.MODEL_FILE}")
        file_data = config.supabase.storage.from_(config.BUCKET_NAME).download(config.MODEL_FILE)
        if not file_data:
            logger.error("[Supabase] Download berhasil tapi data kosong/None")
            return None
        t1 = time.time()
        logger.info(f"[Supabase] Download selesai ({len(file_data)} bytes) dalam {t1-t0:.2f}s")

        try:
            raw_bytes = gzip.decompress(file_data)
            t2 = time.time()
            logger.info(
                f"[Supabase] Gzip decompress {len(file_data)} -> {len(raw_bytes)} bytes "
                f"dalam {t2-t1:.2f}s"
            )
        except OSError:
            # File lama (belum dikompresi) — pakai apa adanya
            raw_bytes = file_data
            t2 = time.time()
            logger.info("[Supabase] File tidak terkompresi (model lama), lanjut tanpa decompress")

        model_data = pickle.loads(raw_bytes)
        t3 = time.time()
        logger.info(
            f"[Supabase] Unpickle dalam {t3-t2:.2f}s "
            f"(total download+load: {t3-t0:.2f}s)"
        )
        return model_data
    except Exception as e:
        logger.error(f"[Supabase] Gagal load model: {type(e).__name__}: {e}")
        return None

def save_model_to_supabase(model_data):
    """Simpan model ke Supabase Storage (dikompresi gzip untuk mempercepat
    download & loading di sisi lain — FIX #7)."""
    if config.supabase is None:
        logger.warning("Supabase not configured, cannot save model")
        return False
    try:
        model_bytes = pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(model_bytes, compresslevel=6)
        logger.info(
            f"[Supabase] Ukuran model: raw={len(model_bytes)} bytes, "
            f"gzip={len(compressed)} bytes "
            f"({100 * len(compressed) / max(len(model_bytes), 1):.0f}% dari ukuran asli)"
        )
        try:
            config.supabase.storage.from_(config.BUCKET_NAME).upload(
                config.MODEL_FILE, compressed,
                file_options={"content-type": "application/octet-stream"}
            )
            logger.info("Model saved to Supabase storage (new file)")
        except Exception:
            config.supabase.storage.from_(config.BUCKET_NAME).update(
                config.MODEL_FILE, compressed,
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
    Synchronous model loading — aman untuk Vercel serverless.
    Tidak pakai background thread / lock karena Vercel membunuh thread
    antar-request sehingga flag bisa stuck selamanya.
    Auto-reset _models_loading jika stuck dari request sebelumnya.
    """
    global model_qa, vectorizer_qa, answers, pertanyaan_list, kategori_list
    global _models_loaded, _models_loading, _X_all_questions_cache

    if _models_loaded:
        return

    # Auto-reset jika flag loading stuck (tidak ada thread aktif di serverless)
    if _models_loading:
        logger.warning("[Model] Flag _models_loading stuck — auto-reset dan retry")
        _models_loading = False

    _models_loading = True
    try:
        model_data = load_model_from_supabase()
        if model_data:
            model_qa       = model_data['model']
            vectorizer_qa  = model_data['vectorizer']
            answers        = model_data['answers']
            pertanyaan_list = model_data['questions']
            kategori_list  = model_data.get('categories', [])
            logger.info(f"[Model] Loaded {len(pertanyaan_list)} questions from Supabase")

            # FIX #8: Pakai TF-IDF matrix yang SUDAH dihitung saat training
            # (disimpan di pickle sebagai 'X_all') alih-alih menghitung ulang
            # preprocess()+transform() untuk seluruh dataset di setiap cold start.
            # Ini biasanya jadi bottleneck terbesar di "load model pertama kali"
            # kalau preprocess() melakukan stemming/normalisasi yang berat.
            cached_X_all = model_data.get('X_all')
            if cached_X_all is not None:
                _X_all_questions_cache = cached_X_all
                logger.info("[Model] TF-IDF matrix dipulihkan langsung dari pickle (skip recompute)")
            else:
                # Backward-compat: model lama belum punya 'X_all', hitung sekali saja.
                # Setelah training ulang berikutnya, jalur ini tidak akan dipakai lagi.
                logger.warning("[Model] Model lama tanpa 'X_all' cache — recompute sekali (lambat)")
                _build_questions_cache()
        else:
            load_dataset_from_db()
            logger.warning("[Model] Tidak ada model di Supabase, dataset saja yang dimuat")
        _models_loaded = True
    except Exception as e:
        logger.error(f"[Model] load_models_and_data error: {e}")
        # Tidak set _models_loaded=True agar bisa retry di request berikutnya
    finally:
        _models_loading = False


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



def predict_svm(user_input):
    """
    Fungsi inti prediksi SVM yang telah diperbaiki untuk menangani kata sapaan pendek (Short-Text Match).
    """
    # 1. Proteksi kata kasar
    if contains_profanity(user_input):
        return {
            'status': 'profanity',
            'jawaban': get_profanity_response(),
            'kategori': 'profanity',
            'confidence_score': 0.0,
            'referensi_alternatif': []
        }

    # 2. Load model & pastikan data tersedia
    load_models_and_data()
    if model_qa is None or vectorizer_qa is None:
        return {
            'status': 'no_model',
            'jawaban': 'Model belum siap, lakukan training terlebih dahulu.',
            'kategori': 'unknown',
            'confidence_score': 0.0,
            'referensi_alternatif': []
        }

    # ==================== FIX: SHORT-TEXT EXACT MATCH ====================
    # Membersihkan input user untuk perbandingan teks langsung (case-insensitive)
    cleaned_user_input = user_input.strip().lower()
    
    # Cek apakah ada kecocokan teks langsung di dalam list pertanyaan database
    for idx, q_db in enumerate(pertanyaan_list):
        if q_db.strip().lower() == cleaned_user_input:
            logger.info(f"[Exact Match] Menemukan kecocokan langsung untuk kata: '{user_input}'")
            return {
                'status': 'ok',
                'jawaban': str(answers[idx]),
                'kategori': str(kategori_list[idx]) if idx < len(kategori_list) else 'umum',
                'confidence_score': 1.0,  # Nilai penuh karena cocok sempurna
                'referensi_alternatif': []
            }
    # =====================================================================

    # 3. Preprocessing & transform input user ke TF-IDF
    processed_input = preprocess(user_input)
    X_input_tfidf = vectorizer_qa.transform([processed_input])

    # Jika kata kunci tidak ada sama sekali di kamus TF-IDF (dan lolos exact match)
    if X_input_tfidf.nnz == 0:
        save_unknown_question(user_input)
        return {
            'status': 'unknown',
            'jawaban': 'Mohon maaf, saya belum mengerti pertanyaan Anda.',
            'kategori': 'unknown',
            'confidence_score': 0.0,
            'referensi_alternatif': []
        }

    # 4. Prediksi kategori menggunakan Linear SVM
    predicted_category = model_qa.predict(X_input_tfidf)[0]

    # 5. Pencocokan jawaban spesifik (mencari dokumen terdekat)
    X_all_questions_tfidf = vectorizer_qa.transform([preprocess(q) for q in pertanyaan_list])
    predict_answer_score = (X_input_tfidf * X_all_questions_tfidf.T).toarray().flatten()

    # Kumpulkan top-3 alternatif global (untuk konteks n8n)
    top_global_indices = sorted(
        range(len(predict_answer_score)),
        key=lambda i: predict_answer_score[i],
        reverse=True
    )[:3]
    referensi_alternatif = [
        {
            'pertanyaan': pertanyaan_list[idx],
            'jawaban': answers[idx],
            'score': float(predict_answer_score[idx])
        }
        for idx in top_global_indices
        if predict_answer_score[idx] > 0.05
    ]

    # Ambil indeks dokumen yang berada di bawah kategori hasil prediksi SVM
    category_indices = [idx for idx, cat in enumerate(kategori_list) if cat == predicted_category]

    if not category_indices:
        save_unknown_question(user_input)
        return {
            'status': 'unknown',
            'jawaban': 'Mohon maaf, saya belum mengerti pertanyaan Anda.',
            'kategori': str(predicted_category),
            'confidence_score': 0.0,
            'referensi_alternatif': referensi_alternatif
        }

    # Cari jawaban dengan skor kecocokan tertinggi di kategori tersebut
    best_index = max(category_indices, key=lambda idx: predict_answer_score[idx])
    max_predict_score = predict_answer_score[best_index]

    # Threshold ketat
    if max_predict_score < 0.15:
        save_unknown_question(user_input)
        return {
            'status': 'unknown',
            'jawaban': 'Mohon maaf, saya belum mengerti pertanyaan Anda.',
            'kategori': str(predicted_category),
            'confidence_score': float(max_predict_score),
            'referensi_alternatif': referensi_alternatif
        }

    # Satu jawaban terbaik
    return {
        'status': 'ok',
        'jawaban': str(answers[best_index]),
        'kategori': str(predicted_category),
        'confidence_score': float(max_predict_score),
        'referensi_alternatif': referensi_alternatif
    }


