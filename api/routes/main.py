import time
from datetime import datetime

from flask import Blueprint, jsonify, render_template

from api import model_service

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_service.model_qa is not None,
        'model_loading': model_service._models_loading,
        'dataset_size': len(model_service.pertanyaan_list)
    })

# FIX #4 (lanjutan): Endpoint warmup yang bisa dipanggil oleh cron/uptime monitor
# agar Vercel instance tidak cold. Panggil endpoint ini setiap 4 menit dari
# layanan cron gratis seperti cron-job.org atau UptimeRobot.
@bp.route('/api/warmup', methods=['GET'])
def warmup():
    """
    Endpoint untuk mencegah cold start Vercel.
    Daftarkan ke UptimeRobot / cron-job.org setiap 4 menit.
    Loading SYNCHRONOUS — selesai dalam satu request, tidak pakai thread.
    """
    t0 = time.time()
    if not model_service._models_loaded:
        model_service.load_models_and_data()
    return jsonify({
        'status': 'ok',
        'model_ready': model_service._models_loaded and model_service.model_qa is not None,
        'dataset_size': len(model_service.pertanyaan_list),
        'elapsed_ms': round((time.time() - t0) * 1000),
        'timestamp': datetime.now().isoformat()
    })


