"""Task 4.4: every setting comes from environment variables, nothing is hard coded."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env when running outside Docker; Compose passes env vars directly


class Settings:
    def __init__(self):
        self.inference_mode = os.getenv("INFERENCE_MODE", "local")  # local | remote
        self.model_path = os.getenv("MODEL_PATH", "models/champion.keras")
        self.model_version = os.getenv("MODEL_VERSION", "champion")
        self.threshold = float(os.getenv("THRESHOLD", "0.5"))
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
        self.max_upload_mb = float(os.getenv("MAX_UPLOAD_MB", "5"))
        self.model_api_url = os.getenv("MODEL_API_URL", "")
        self.model_api_key = os.getenv("MODEL_API_KEY", "")
        self.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(16).hex())

    def safe_dict(self):
        """Settings that are safe to show on the health page (no passwords)."""
        db = self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url
        return {"inference_mode": self.inference_mode, "model_path": self.model_path,
                "threshold": self.threshold, "database": db, "max_upload_mb": self.max_upload_mb,
                "model_api_url": self.model_api_url or None}
