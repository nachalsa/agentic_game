import re
import json
import logging
import threading
from typing import List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from config import config

logger = logging.getLogger(__name__)

# ===== 입력 검증 =====
class InputValidator:
    """사용자 입력 검증 및 정제"""
    
    DANGEROUS_PATTERNS = [
        r'<script.*?</script>',
        r'javascript:',
        r'eval\s*\(',
        r'<.*?>',  # HTML 태그 제거
    ]
    
    @classmethod
    def sanitize_input(cls, user_input: str) -> str:
        """사용자 입력 정제"""
        if not user_input or not user_input.strip():
            return ""
        
        # 길이 제한
        if len(user_input) > config.MAX_INPUT_LENGTH:
            user_input = user_input[:config.MAX_INPUT_LENGTH]
            logger.warning(f"입력이 최대 길이로 제한됨: {config.MAX_INPUT_LENGTH}자")
        
        # 위험한 패턴 제거
        for pattern in cls.DANGEROUS_PATTERNS:
            user_input = re.sub(pattern, '', user_input, flags=re.IGNORECASE)
        
        return user_input.strip()
    
    @classmethod
    def validate_command(cls, command: str) -> bool:
        """명령어 유효성 검사"""
        valid_commands = {
            'help', 'quit', 'save', 'load', 'status', 'inventory',
            '조사', '수락', '거절', '공격', '도망', '상태', '인벤토리'
        }
        
        # 기본 명령어이거나 자유 입력 허용
        return command.lower() in valid_commands or len(command.split()) <= 20

# ===== 데이터 모델 =====
@dataclass
class Character:
    name: str
    level: int = 1
    hp: int = 10
    max_hp: int = 10
    ac: int = 10
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    inventory: List[str] = None
    
    def __post_init__(self):
        if self.inventory is None:
            self.inventory = ["기본 검", "가죽 갑옷", "체력 물약 x2"]
    
    def get_ability_modifier(self, ability_score: int) -> int:
        """능력치 수정치 계산"""
        return (ability_score - 10) // 2
    
    def is_alive(self) -> bool:
        """생존 여부 확인"""
        return self.hp > 0
    
    def heal(self, amount: int):
        """체력 회복"""
        self.hp = min(self.max_hp, self.hp + amount)
    
    def take_damage(self, damage: int):
        """피해 입기"""
        self.hp = max(0, self.hp - damage)

@dataclass
class GameState:
    current_scene: str = "시작 지점"
    active_characters: List[Character] = None
    session_log: List[str] = None
    turn_order: List[str] = None
    game_context: str = ""
    created_at: str = None
    last_updated: str = None
    
    def __post_init__(self):
        if self.active_characters is None:
            self.active_characters = []
        if self.session_log is None:
            self.session_log = []
        if self.turn_order is None:
            self.turn_order = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

# ===== 게임 상태 매니저 =====
class GameStateManager:
    """Thread-safe 싱글톤 게임 상태 매니저"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.state = GameState()
                    cls._instance.saves_dir = Path("saves")
                    cls._instance.saves_dir.mkdir(exist_ok=True)
        return cls._instance
    
    def update_context(self, new_context: str):
        """게임 컨텍스트 업데이트"""
        with self._lock:
            self.state.game_context = new_context
            self.state.session_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {new_context}")
            self.state.last_updated = datetime.now().isoformat()
            logger.info(f"게임 컨텍스트 업데이트: {new_context[:100]}...")
    
    def get_context(self) -> str:
        """현재 게임 컨텍스트 조회"""
        return self.state.game_context
    
    def add_character(self, character: Character):
        """캐릭터 추가"""
        with self._lock:
            self.state.active_characters.append(character)
            logger.info(f"캐릭터 추가됨: {character.name}")
    
    def get_character(self, name: str) -> Optional[Character]:
        """캐릭터 조회"""
        for char in self.state.active_characters:
            if char.name.lower() == name.lower():
                return char
        return None
    
    def save_game(self, filename: str = None) -> bool:
        """게임 상태 저장"""
        try:
            if filename is None:
                filename = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            save_path = self.saves_dir / filename
            
            with self._lock:
                save_data = {
                    **asdict(self.state),
                    'save_timestamp': datetime.now().isoformat()
                }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"게임 저장 완료: {save_path}")
            return True
        except Exception as e:
            logger.error(f"게임 저장 실패: {e}")
            return False
    
    def load_game(self, filename: str) -> bool:
        """게임 상태 불러오기"""
        try:
            save_path = self.saves_dir / filename
            if not save_path.exists():
                logger.warning(f"저장 파일을 찾을 수 없음: {save_path}")
                return False
            
            with open(save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # GameState 복원
            with self._lock:
                characters_data = data.pop('active_characters', [])
                data.pop('save_timestamp', None)  # 저장 시간은 제외
                
                self.state = GameState(**data)
                
                # 캐릭터 객체 복원
                self.state.active_characters = [
                    Character(**char_data) for char_data in characters_data
                ]
            
            logger.info(f"게임 불러오기 완료: {save_path}")
            return True
        except Exception as e:
            logger.error(f"게임 불러오기 실패: {e}")
            return False
    
    def get_save_files(self) -> List[str]:
        """저장 파일 목록 조회"""
        try:
            return [f.name for f in self.saves_dir.glob("*.json")]
        except Exception as e:
            logger.error(f"저장 파일 목록 조회 실패: {e}")
            return []

# 글로벌 게임 상태 매니저
game_state_manager = GameStateManager()