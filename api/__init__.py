import logging
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from api import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def create_app():
    base_dir = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=str(base_dir / 'static'),
    )

    CORS(
        app,
        origins=config.ALLOWED_ORIGINS,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Requested-With"],
        supports_credentials=True,
        max_age=86400,
    )

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

    from api.routes.main import bp as main_bp
    from api.routes.chat import bp as chat_bp
    from api.routes.auth_admin import bp as auth_admin_bp
    from api.routes.data import bp as data_bp
    from api.routes.debug import bp as debug_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(auth_admin_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(debug_bp)

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e) if config.FLASK_DEBUG else 'Terjadi kesalahan',
        }), 500

    if config.STARTUP_WARMUP:
        from api import model_service

        model_service.warmup_models_async()
        logger.info("[App] Siap. Warmup model berjalan di background.")
    else:
        logger.info("[App] Siap. Model akan dimuat synchronous saat request pertama.")
    return app
