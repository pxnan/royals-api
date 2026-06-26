import logging

from flask import Blueprint, jsonify, request

from api import model_service
from api.auth import api_key_required
from api.database import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('chat', __name__)


@bp.route('/api/chat-n8n-proxy', methods=['POST', 'OPTIONS'])
@api_key_required
def chat_n8n_proxy():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json
    if not data or 'pertanyaan' not in data:
        return jsonify({'error': 'Pertanyaan tidak boleh kosong'}), 400

    user_input = data['pertanyaan']
    logger.info(f"[claude-proxy] Memproses: {user_input}")

    # Pakai fungsi prediksi yang sama dengan /api/chat
    result = model_service.predict_svm(user_input)

    if result['status'] == 'no_model':
        return jsonify({'error': result['jawaban']}), 500

    jawaban_svm = result['jawaban']
    kategori    = result['kategori']

    # Susun referensi alternatif sebagai teks untuk konteks Claude
    lines = [
        f"Alternatif {i+1}:\nPertanyaan: {r['pertanyaan']}\nJawaban: {r['jawaban']}"
        for i, r in enumerate(result['referensi_alternatif'])
    ]
    referensi_alternatif_text = "\n\n".join(lines) if lines else "TIDAK_ADA_ALTERNATIF"

    # Prompt yang identik dengan yang digunakan di n8n QA & Enhancer Agent
    user_prompt = f"""Anda adalah AI Quality Assurance sekaligus Enhancer bahasa untuk Chatbot Royal's Resto.
Tugas Anda adalah memvalidasi, mengoreksi, atau memoles jawaban klasifikasi SVM agar menjadi sangat ramah dan profesional khas restoran bintang lima.

Input Utama:
- PERTANYAAN USER: {user_input}
- JAWABAN DASAR SVM: {jawaban_svm}
- KATEGORI TERPREDIKSI: {kategori}

Data Referensi Cadangan (Top Terkait):
{referensi_alternatif_text}

Instruksi Logika Evaluasi & Pengambilan Keputusan:
1. JIKA JAWABAN DASAR SVM sudah relevan, akurat, dan menjawab PERTANYAAN USER dengan benar (KATEGORI bukan "unknown"):
   Gunakan informasi penting dari JAWABAN DASAR SVM tersebut (Jangan pernah mengubah data penting seperti harga atau nama menu asli!). Poles susunan kalimatnya agar ramah. Jika teks mengandung format kurung kurawal '{{}}' atau tanda pipa '|', hilangkan tanda tersebut dan susun menjadi daftar list bullet points markdown (* ) berjejer rapi ke bawah.
2. JIKA KATEGORI bernilai "unknown" ATAU Anda merasa JAWABAN DASAR SVM tidak nyambung dengan PERTANYAAN USER:
   Abaikan JAWABAN DASAR SVM tersebut. Lihat bagian "Data Referensi Cadangan (Top Terkait)" di atas.
   - Jika Anda melihat ada "Alternatif" pertanyaan yang dirasa memiliki makna yang cocok dan benar maksudnya dengan apa yang ditanyakan USER, ambil data "Jawaban" dari alternatif tersebut, lalu poles kalimatnya agar ramah dan komunikatif.
   - Jika tidak ada alternatif yang cocok atau saat kamu cek dan hasilnya meragukan dengan apa maksud yang ditanyakan user, atau isinya "TIDAK_ADA_ALTERNATIF", berikan respon penolakan baru yang sangat sopan dan ramah. Katakan bahwa Royal's Resto belum menyediakan informasi mendetail mengenai hal tersebut saat ini. Kemudian, sarankan secara halus kepada pelanggan untuk bertanya seputar menu makanan spesial, lokasi restoran, jam operasional, atau bantuan melakukan reservasi meja.

Aturan Tambahan:
- Langsung berikan hasil jawaban akhir yang siap dibaca oleh pelanggan tanpa menyertakan kata pengantar buatan AI seperti "Berikut hasil polesan saya:" atau "Berdasarkan alternatif yang saya temukan..."."""

    system_message = "You are a professional restaurant assistant for Royal's Resto, a five-star establishment. Your role is to validate and enhance chatbot responses to be warm, friendly, and professional. Always maintain the accuracy of factual information like prices and menu names while improving the conversational tone."

    try:
        import requests as req_lib
        alibaba_url = "https://ws-fvkye3i925gbopy6.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages"
        headers_alibaba = {
            "Content-Type": "application/json",
            "x-api-key": "sk-ws-H.LDDLRD.1amw.MEYCIQCfYqXfwZGkTNdFSwbVKV8buO7vzEgcPn8H-Yb_J-t_rgIhAId5Uv7R5oDHMKI4rXjpp55tQRLLumiWkcQSBrme-kNn",
            "anthropic-version": "2023-06-01"
        }
        payload_alibaba = {
            "model": "deepseek-v3.2",
            "max_tokens": 1024,
            "system": system_message,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        response_alibaba = req_lib.post(alibaba_url, json=payload_alibaba, headers=headers_alibaba, timeout=30)

        if response_alibaba.status_code == 200:
            alibaba_data = response_alibaba.json()
            answer_text = alibaba_data.get("content", [{}])[0].get("text", jawaban_svm)
            return jsonify({'answer': answer_text, 'status': 'success'}), 200
        else:
            logger.error(f"[alibaba-proxy] Alibaba MaaS API error {response_alibaba.status_code}: {response_alibaba.text}")
            return jsonify({'status': 'success', 'answer': jawaban_svm}), 200

    except Exception as e:
        logger.error(f"[alibaba-proxy] Koneksi Alibaba MaaS gagal, mengaktifkan fallback: {str(e)}")
        return jsonify({'status': 'success', 'answer': jawaban_svm}), 200


@bp.route('/api/chat/ambiguous-unknown', methods=['POST', 'OPTIONS'])
@api_key_required
def handle_ambiguous_unknown():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json or {}
    original_question = data.get('original_question', '').strip()

    if not original_question:
        return jsonify({'error': 'Pertanyaan asli tidak ditemukan'}), 400

    if model_service.contains_profanity(original_question):
        return jsonify({
            'status': 'error',
            'jawaban': model_service.get_profanity_response(),
            'original_question': original_question
        }), 200

    model_service.save_unknown_question(original_question)

    return jsonify({
        'status': 'unknown',
        'jawaban': 'Mohon maaf, saya belum mengerti pertanyaan Anda. Pertanyaan Anda telah saya catat untuk pembelajaran ke depannya.',
        'original_question': original_question
    }), 200


# ==================== RECOMMENDATIONS ====================
@bp.route('/api/recommendations/random', methods=['GET', 'OPTIONS'])
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

@bp.route('/api/recommendations/by-category', methods=['GET', 'OPTIONS'])
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

