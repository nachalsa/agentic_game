import os
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import random
import json
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv
import re
from contextlib import contextmanager

# .env 파일 로드
load_dotenv()

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
logger = logging.getLogger(__name__)

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
        logger.info(f"LiteLLM 설정 완료 - Model: {self.MODEL_NAME}, URL: {self.API_BASE_URL}")

# 글로벌 설정 인스턴스
config = GameConfig()

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

# ===== 개선된 게임 상태 매니저 =====
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

# ===== 개선된 도구들 =====
class DiceRollInput(BaseModel):
    sides: int = Field(default=20, description="주사위 면 수", ge=2, le=100)
    count: int = Field(default=1, description="주사위 개수", ge=1, le=10)
    modifier: int = Field(default=0, description="수정치", ge=-20, le=20)

class DiceRollTool(BaseTool):
    name: str = "roll_dice"
    description: str = "주사위를 굴립니다 (2d6+3 형태로 입력)"
    args_schema: type[BaseModel] = DiceRollInput
    
    def _run(self, sides: int = 20, count: int = 1, modifier: int = 0) -> str:
        try:
            rolls = [random.randint(1, sides) for _ in range(count)]
            total = sum(rolls) + modifier
            
            result = {
                "rolls": rolls,
                "modifier": modifier,
                "total": total,
                "description": f"{count}d{sides}+{modifier} = {rolls} + {modifier} = {total}",
                "critical": any(roll == sides for roll in rolls),
                "fumble": any(roll == 1 for roll in rolls) and sides == 20
            }
            
            logger.info(f"주사위 굴림: {result['description']}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"주사위 굴리기 실패: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

class AbilityCheckInput(BaseModel):
    ability_score: int = Field(description="능력치 수치", ge=1, le=30)
    difficulty: int = Field(default=10, description="난이도", ge=5, le=30)
    advantage: bool = Field(default=False, description="유리함 여부")
    disadvantage: bool = Field(default=False, description="불리함 여부")

class AbilityCheckTool(BaseTool):
    name: str = "ability_check"
    description: str = "능력치 판정을 수행합니다"
    args_schema: type[BaseModel] = AbilityCheckInput
    
    def _run(self, ability_score: int, difficulty: int = 10, advantage: bool = False, disadvantage: bool = False) -> str:
        try:
            # 유리함/불리함 처리
            if advantage and disadvantage:
                advantage = disadvantage = False  # 상쇄
            
            if advantage:
                roll1, roll2 = random.randint(1, 20), random.randint(1, 20)
                roll = max(roll1, roll2)
                roll_desc = f"2d20 유리함({roll1}, {roll2}) -> {roll}"
            elif disadvantage:
                roll1, roll2 = random.randint(1, 20), random.randint(1, 20)
                roll = min(roll1, roll2)
                roll_desc = f"2d20 불리함({roll1}, {roll2}) -> {roll}"
            else:
                roll = random.randint(1, 20)
                roll_desc = f"d20({roll})"
            
            modifier = (ability_score - 10) // 2
            total = roll + modifier
            success = total >= difficulty
            
            result = {
                "roll": roll,
                "modifier": modifier,
                "total": total,
                "difficulty": difficulty,
                "success": success,
                "critical_success": roll == 20,
                "critical_failure": roll == 1,
                "description": f"{roll_desc} + 수정치({modifier}) = {total} vs DC{difficulty} - {'성공' if success else '실패'}"
            }
            
            logger.info(f"능력치 판정: {result['description']}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"능력치 판정 실패: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

class GameContextTool(BaseTool):
    name: str = "get_game_context"
    description: str = "현재 게임 상황과 컨텍스트를 가져옵니다"
    
    def _run(self) -> str:
        try:
            context = game_state_manager.get_context()
            characters = [asdict(char) for char in game_state_manager.state.active_characters]
            
            result = {
                "current_context": context,
                "characters": characters,
                "scene": game_state_manager.state.current_scene,
                "last_updated": game_state_manager.state.last_updated
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"게임 컨텍스트 조회 실패: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

class UpdateContextInput(BaseModel):
    new_context: str = Field(description="새로운 게임 컨텍스트")

class UpdateContextTool(BaseTool):
    name: str = "update_game_context"
    description: str = "게임 상황과 컨텍스트를 업데이트합니다"
    args_schema: type[BaseModel] = UpdateContextInput
    
    def _run(self, new_context: str) -> str:
        try:
            game_state_manager.update_context(new_context)
            return f"✅ 게임 컨텍스트가 업데이트되었습니다: {new_context[:100]}..."
        except Exception as e:
            logger.error(f"게임 컨텍스트 업데이트 실패: {e}")
            return f"❌ 컨텍스트 업데이트 실패: {str(e)}"

# Tool 인스턴스 생성
dice_tool = DiceRollTool()
ability_tool = AbilityCheckTool()
context_tool = GameContextTool()
update_context_tool = UpdateContextTool()

# ===== Agent 정의 =====
def create_agents():
    """에이전트 생성"""
    try:
        # Game Master Agent
        game_master = Agent(
            role="게임 마스터",
            goal="플레이어 입력에 따라 즉시 반응하고 재미있는 게임을 진행",
            backstory="""당신은 숙련된 D&D 게임 마스터입니다. 
            플레이어의 행동에 즉시 반응하고 흥미진진한 상황을 만들어냅니다.
            필요시 주사위를 굴리고 상황을 업데이트합니다.""",
            tools=[dice_tool, context_tool, update_context_tool],
            verbose=True,
            llm=f"openai/{config.MODEL_NAME}",
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE
        )

        # Rules Advisor Agent
        rules_advisor = Agent(
            role="규칙 조언자",
            goal="복잡한 상황에서 D&D 규칙 조언 제공",
            backstory="""D&D 5판 규칙 전문가로서, 복잡한 상황에서만 
            규칙 해석과 판정 조언을 제공합니다.""",
            tools=[ability_tool, dice_tool],
            verbose=True,
            llm=f"openai/{config.MODEL_NAME}",
            max_tokens=config.MAX_TOKENS // 2,
            temperature=0.3
        )
        
        logger.info("에이전트 생성 완료")
        return game_master, rules_advisor
    except Exception as e:
        logger.error(f"에이전트 생성 실패: {e}")
        raise

# ===== 개선된 게임 엔진 =====
class ImprovedDnDGameEngine:
    """개선된 D&D 게임 엔진"""
    
    def __init__(self):
        self.is_running = False
        self.offline_mode = False
        self._crew = None
        self._agents = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def test_connection(self) -> bool:
        """LLM 연결 테스트"""
        try:
            self.logger.info("LLM 연결 테스트 중...")
            response = litellm.completion(
                model=f"openai/{config.MODEL_NAME}",
                messages=[{"role": "user", "content": "연결 테스트"}],
                api_base=config.API_BASE_URL,
                api_key=config.API_KEY,
                temperature=0.1,
                max_tokens=50,
                timeout=config.TIMEOUT
            )
            self.logger.info("✅ LLM 연결 성공!")
            return True
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"❌ 연결 오류: {e}")
            self.offline_mode = True
            return False
        except requests.exceptions.Timeout as e:
            self.logger.error(f"❌ 시간 초과: {e}")
            self.offline_mode = True
            return False
        except litellm.AuthenticationError as e:
            self.logger.error(f"❌ 인증 오류: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 예상치 못한 오류: {e}")
            self.offline_mode = True
            return False
    
    def _get_crew(self):
        """Crew 인스턴스 가져오기 (재사용)"""
        if self._crew is None or self._agents is None:
            try:
                self._agents = create_agents()
                self._crew = Crew(
                    agents=self._agents,
                    tasks=[],  # 동적으로 추가
                    process=Process.sequential,
                    verbose=False
                )
                self.logger.info("Crew 인스턴스 생성 완료")
            except Exception as e:
                self.logger.error(f"Crew 생성 실패: {e}")
                raise
        return self._crew
    
    @contextmanager
    def _error_handler(self, operation: str):
        """에러 처리 컨텍스트 매니저"""
        try:
            yield
        except requests.exceptions.ConnectionError as e:
            self.logger.warning(f"{operation} - 연결 오류: {e}")
            self.offline_mode = True
        except requests.exceptions.Timeout as e:
            self.logger.warning(f"{operation} - 시간 초과: {e}")
            self.offline_mode = True
        except litellm.AuthenticationError as e:
            self.logger.error(f"{operation} - 인증 오류: {e}")
            raise
        except Exception as e:
            self.logger.error(f"{operation} - 예상치 못한 오류: {e}")
            raise
    
    def start_game(self) -> str:
        """게임 시작"""
        self.logger.info("=== D&D Crew AI 게임 시작 ===")
        
        # 연결 테스트
        if not self.test_connection():
            self.logger.warning("오프라인 모드로 시작")
        
        self.is_running = True
        
        # 기본 캐릭터 생성
        default_character = Character(
            name="모험가",
            strength=14, dexterity=12, constitution=13,
            intelligence=10, wisdom=11, charisma=9
        )
        game_state_manager.add_character(default_character)
        
        # 시작 시나리오
        start_scenario = """
🏰 **모험의 시작**

당신은 작은 마을 '그린힐'의 여관 '황금 용'에 있습니다. 
여관 주인이 다가와 말합니다:

"모험가님, 마을 근처 동굴에서 이상한 소리가 들린다는 신고가 있었습니다. 
조사해주시면 50골드를 드리겠습니다."

**현재 상태:**
- 체력: 10/10
- 장비: 기본 검, 가죽 갑옷, 체력 물약 x2
- 위치: 그린힐 마을 여관

무엇을 하시겠습니까?
        """
        
        game_state_manager.update_context(start_scenario.strip())
        return start_scenario.strip()
    
    def process_input(self, player_input: str) -> str:
        """플레이어 입력 처리"""
        if not self.is_running:
            return "❌ 게임이 시작되지 않았습니다. start_game()을 먼저 호출하세요."
        
        # 입력 검증
        sanitized_input = InputValidator.sanitize_input(player_input)
        if not sanitized_input:
            return "❌ 유효하지 않은 입력입니다."
        
        if not InputValidator.validate_command(sanitized_input):
            return "❌ 입력이 너무 깁니다. 간단하게 입력해주세요."
        
        self.logger.info(f"플레이어 입력: {sanitized_input}")
        
        if self.offline_mode:
            return self._offline_response(sanitized_input)
        
        with self._error_handler("플레이어 입력 처리"):
            try:
                # 동적 Task 생성
                response_task = Task(
                    description=f"""
                    플레이어 액션: "{sanitized_input}"
                    
                    현재 게임 상황을 파악하고 플레이어의 행동에 즉시 반응하세요.
                    필요하면 주사위를 굴리고, 결과에 따라 새로운 상황을 묘사하세요.
                    게임 컨텍스트를 업데이트하는 것도 잊지 마세요.
                    
                    응답 형식:
                    - 행동 결과 묘사
                    - 필요시 주사위 결과
                    - 새로운 상황이나 선택지 제시
                    
                    한국어로 자연스럽고 재미있게 응답하세요.
                    """,
                    agent=self._get_crew().agents[0],  # game_master
                    expected_output="플레이어 행동에 대한 즉각적인 반응과 새로운 상황"
                )
                
                # 새로운 Task로 Crew 업데이트
                crew = self._get_crew()
                crew.tasks = [response_task]
                
                result = crew.kickoff()
                self.logger.info("GM 응답 생성 완료")
                return str(result)
                
            except Exception as e:
                self.logger.error(f"입력 처리 중 오류: {e}")
                return self._offline_response(sanitized_input)
    
    def _offline_response(self, player_input: str) -> str:
        """오프라인 모드 응답"""
        self.logger.info(f"오프라인 모드로 응답: {player_input}")
        
        responses = {
            "조사": "🔍 동굴을 조사하기로 했습니다. 어둠 속에서 이상한 소리가 들립니다...",
            "수락": "✅ 퀘스트를 수락했습니다! 동굴 입구로 향합니다.",
            "거절": "❌ 여관 주인이 실망한 표정을 짓습니다. 다른 모험가를 찾아보겠다고 합니다.",
            "인벤토리": "🎒 현재 소지품: 기본 검, 가죽 갑옷, 체력 물약 x2",
            "상태": "❤️ 체력: 10/10, 컨디션: 양호",
            "도움": "📖 사용 가능한 명령어: 조사, 수락, 거절, 인벤토리, 상태, 공격, 도망"
        }
        
        for key, response in responses.items():
            if key in player_input.lower():
                return f"🎭 GM: {response}"
        
        # 기본 응답
        return f"🎭 GM: '{player_input}'에 대해 생각해봅니다... (오프라인 모드 - LLM 서버 연결을 확인해주세요)"
    
    def get_help(self) -> str:
        """도움말"""
        return """
📖 **D&D 게임 명령어:**

**기본 명령어:**
- 'help' - 이 도움말 표시
- 'quit' - 게임 종료
- 'save [파일명]' - 게임 저장
- 'load [파일명]' - 게임 불러오기
- 'saves' - 저장 파일 목록

**게임 내 행동:**
- '조사하기' - 주변을 조사
- '인벤토리' - 소지품 확인  
- '상태' - 캐릭터 상태 확인
- '수락/거절' - 퀘스트 수락/거절
- '공격하기' - 전투 시작
- '도망가기' - 전투에서 도망
- '말하기: [내용]' - NPC와 대화
- '마법 사용: [마법명]' - 마법 시전
- '아이템 사용: [아이템명]' - 아이템 사용

**시스템 정보:**
- 최대 입력 길이: {config.MAX_INPUT_LENGTH}자
- 오프라인 모드: {'활성' if self.offline_mode else '비활성'}
- LLM 모델: {config.MODEL_NAME}

자유롭게 행동을 입력하세요! 게임이 자연스럽게 반응합니다.
        """
    
    def save_game(self, filename: str = None) -> str:
        """게임 저장"""
        try:
            if game_state_manager.save_game(filename):
                actual_filename = filename or f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                return f"💾 게임이 저장되었습니다: {actual_filename}"
            else:
                return "❌ 게임 저장에 실패했습니다."
        except Exception as e:
            self.logger.error(f"게임 저장 오류: {e}")
            return f"❌ 게임 저장 중 오류가 발생했습니다: {str(e)}"
    
    def load_game(self, filename: str) -> str:
        """게임 불러오기"""
        try:
            if game_state_manager.load_game(filename):
                return f"📁 게임을 불러왔습니다: {filename}\n\n{game_state_manager.get_context()}"
            else:
                return f"❌ 저장 파일을 찾을 수 없습니다: {filename}"
        except Exception as e:
            self.logger.error(f"게임 불러오기 오류: {e}")
            return f"❌ 게임 불러오기 중 오류가 발생했습니다: {str(e)}"
    
    def list_saves(self) -> str:
        """저장 파일 목록"""
        try:
            saves = game_state_manager.get_save_files()
            if not saves:
                return "📂 저장된 게임이 없습니다."
            
            save_list = "\n".join([f"  - {save}" for save in sorted(saves, reverse=True)])
            return f"📂 **저장된 게임 목록:**\n{save_list}\n\n사용법: load [파일명]"
        except Exception as e:
            self.logger.error(f"저장 파일 목록 조회 오류: {e}")
            return f"❌ 저장 파일 목록을 불러올 수 없습니다: {str(e)}"
    
    def get_status(self) -> str:
        """게임 상태 정보"""
        try:
            characters = game_state_manager.state.active_characters
            if not characters:
                return "❌ 활성 캐릭터가 없습니다."
            
            status_info = []
            for char in characters:
                status_info.append(f"""
**{char.name}** (레벨 {char.level})
- ❤️ 체력: {char.hp}/{char.max_hp}
- 🛡️ 방어도: {char.ac}
- 💪 힘: {char.strength} ({char.get_ability_modifier(char.strength):+d})
- 🏃 민첩: {char.dexterity} ({char.get_ability_modifier(char.dexterity):+d})
- 🛡️ 체질: {char.constitution} ({char.get_ability_modifier(char.constitution):+d})
- 🧠 지능: {char.intelligence} ({char.get_ability_modifier(char.intelligence):+d})
- 🦉 지혜: {char.wisdom} ({char.get_ability_modifier(char.wisdom):+d})
- 💬 매력: {char.charisma} ({char.get_ability_modifier(char.charisma):+d})
- 🎒 인벤토리: {', '.join(char.inventory)}
                """.strip())
            
            return "\n\n".join(status_info)
        except Exception as e:
            self.logger.error(f"상태 조회 오류: {e}")
            return f"❌ 상태를 조회할 수 없습니다: {str(e)}"

# ===== 개선된 메인 실행 함수 =====
def run_game():
    """개선된 게임 실행"""
    try:
        print("🎲 D&D Crew AI 게임 엔진을 초기화하는 중...")
        
        # 게임 엔진 생성
        game = ImprovedDnDGameEngine()
        
        # 게임 시작
        start_message = game.start_game()
        print(start_message)
        
        print("\n" + "="*60)
        print("🎮 게임이 시작되었습니다!")
        print("'help'를 입력하면 명령어를 볼 수 있습니다.")
        print("'quit'를 입력하면 게임을 종료합니다.")
        print("="*60)
        
        # 게임 루프
        while game.is_running:
            try:
                user_input = input("\n🎲 > ").strip()
                
                if not user_input:
                    print("💭 무엇을 하시겠습니까?")
                    continue
                
                # 기본 명령어 처리
                if user_input.lower() == 'quit':
                    print("🎮 게임을 종료합니다. 즐거운 모험이었습니다!")
                    break
                elif user_input.lower() == 'help':
                    print(game.get_help())
                elif user_input.lower() == 'status':
                    print(game.get_status())
                elif user_input.lower() == 'saves':
                    print(game.list_saves())
                elif user_input.lower().startswith('save'):
                    # save 또는 save filename 형태
                    parts = user_input.split(' ', 1)
                    filename = parts[1] if len(parts) > 1 else None
                    print(game.save_game(filename))
                elif user_input.lower().startswith('load'):
                    # load filename 형태
                    parts = user_input.split(' ', 1)
                    if len(parts) < 2:
                        print("❌ 파일명을 입력해주세요. 예: load save_20240101_120000.json")
                        print(game.list_saves())
                    else:
                        filename = parts[1]
                        print(game.load_game(filename))
                else:
                    # 일반 게임 입력 처리
                    print("🎭 GM이 생각하는 중...")
                    response = game.process_input(user_input)
                    print(f"\n{response}")
                    
            except KeyboardInterrupt:
                print("\n\n🎮 게임을 종료합니다.")
                break
            except EOFError:
                print("\n\n🎮 입력 스트림이 종료되었습니다.")
                break
            except Exception as e:
                logger.error(f"게임 루프 중 오류: {e}")
                print(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
                print("게임을 계속 진행합니다...")
                
    except KeyboardInterrupt:
        print("\n🎮 게임 초기화가 중단되었습니다.")
    except Exception as e:
        logger.error(f"게임 실행 중 치명적 오류: {e}")
        print(f"❌ 게임 실행 중 치명적 오류가 발생했습니다: {str(e)}")
        print("\n🔧 문제 해결 방법:")
        print("1. .env 파일의 환경 변수 확인")
        print("   - DEFAULT_LLM: LLM 모델명")
        print("   - DEFAULT_URL: API 서버 URL")
        print("   - DEFAULT_API_KEY: API 키")
        print("2. VLLM/LLM 서버 상태 확인")
        print("3. 의존성 패키지 재설치: pip install -r requirements.txt")
        print("4. 로그 파일 확인: logs/ 디렉터리")

# ===== 추가 유틸리티 함수 =====
def create_character_interactive() -> Character:
    """대화형 캐릭터 생성"""
    print("\n🧙‍♂️ **새 캐릭터 생성**")
    
    name = input("캐릭터 이름: ").strip()
    if not name:
        name = "모험가"
    
    print("\n능력치를 분배하세요 (총 72포인트, 각 능력치 8-18):")
    abilities = {}
    total_points = 72
    base_score = 8
    
    for ability in ['strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma']:
        korean_names = {
            'strength': '힘', 'dexterity': '민첩', 'constitution': '체질',
            'intelligence': '지능', 'wisdom': '지혜', 'charisma': '매력'
        }
        
        while True:
            try:
                score = int(input(f"{korean_names[ability]} (8-18, 남은 포인트: {total_points}): "))
                if 8 <= score <= 18 and score <= total_points:
                    abilities[ability] = score
                    total_points -= (score - base_score)
                    break
                else:
                    print(f"8-18 사이의 값을 입력하고, 남은 포인트({total_points})를 초과하지 마세요.")
            except ValueError:
                print("숫자를 입력해주세요.")
    
    # 체력 계산
    con_modifier = (abilities['constitution'] - 10) // 2
    max_hp = 10 + con_modifier
    
    character = Character(
        name=name,
        hp=max_hp,
        max_hp=max_hp,
        **abilities
    )
    
    print(f"\n✅ {name} 캐릭터가 생성되었습니다!")
    return character

def show_welcome():
    """환영 메시지 표시"""
    welcome_art = """
    ╔═══════════════════════════════════════╗
    ║       🎲 D&D Crew AI 게임 엔진        ║
    ║                                       ║
    ║     AI 게임 마스터와 함께하는         ║
    ║        판타지 롤플레잉 게임           ║
    ║                                       ║
    ║  Powered by CrewAI & LiteLLM          ║
    ╚═══════════════════════════════════════╝
    """
    
    print(welcome_art)
    print("\n🌟 개선된 기능들:")
    print("  • 스레드 세이프 게임 상태 관리")
    print("  • 향상된 에러 처리 및 로깅")
    print("  • 게임 저장/불러오기 시스템")
    print("  • 입력 검증 및 보안 강화")
    print("  • 오프라인 모드 지원")
    print("  • 상세한 캐릭터 상태 관리")
    print("\n게임을 시작하려면 Enter를 누르세요...")
    input()

if __name__ == "__main__":
    try:
        show_welcome()
        run_game()
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        print(f"❌ 프로그램을 시작할 수 없습니다: {str(e)}")
    finally:
        print("\n👋 D&D Crew AI를 이용해주셔서 감사합니다!")