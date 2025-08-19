import os
import logging
import litellm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 크루 AI 텔레메트리 비활성화
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# ===== 로깅 설정 =====
def setup_logging():
    """로깅 시스템 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'dnd_game_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

setup_logging()

# ===== 설정 관리 =====
class GameConfig:
    """게임 설정 관리 클래스"""
    
    def __init__(self):
        self.MODEL_NAME = os.getenv("DEFAULT_LLM", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
        self.API_BASE_URL = self._normalize_url(os.getenv("DEFAULT_URL", "http://localhost:54321"))
        self.API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.TIMEOUT = int(os.getenv("TIMEOUT", "30"))
        self.MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "500"))
        
        # 설정 유효성 검사
        self.validate()
        
        # LiteLLM 글로벌 설정
        self._setup_litellm()
    
    def _normalize_url(self, url: str) -> str:
        """URL 정규화"""
        if not url:
            raise ValueError("API_BASE_URL이 설정되지 않았습니다.")
        return url.rstrip('/') + '/v1' if not url.endswith('/v1') else url
    
    def validate(self):
        """설정 유효성 검사"""
        if not self.API_KEY:
            raise ValueError("API_KEY가 설정되지 않았습니다.")
        if not self.API_BASE_URL:
            raise ValueError("API_BASE_URL이 설정되지 않았습니다.")
        if self.MAX_TOKENS <= 0:
            raise ValueError("MAX_TOKENS는 0보다 커야 합니다.")
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError("TEMPERATURE는 0.0과 2.0 사이여야 합니다.")
    
    def _setup_litellm(self):
        """LiteLLM 설정"""
        litellm.api_base = self.API_BASE_URL
        litellm.api_key = self.API_KEY
        litellm.drop_params = True
        logger = logging.getLogger(__name__)
        logger.info(f"LiteLLM 설정 완료 - Model: {self.MODEL_NAME}, URL: {self.API_BASE_URL}")

# 글로벌 설정 인스턴스
config = GameConfig()