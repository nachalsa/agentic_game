import os
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
MODEL_NAME = os.getenv("DEFAULT_LLM", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://localhost:54321")
API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")

# URL 정규화
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# LiteLLM 글로벌 설정
litellm.api_base = API_BASE_URL
litellm.api_key = API_KEY
litellm.drop_params = True

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

@dataclass
class GameState:
    current_scene: str = "시작 지점"
    active_characters: List[Character] = None
    session_log: List[str] = None
    turn_order: List[str] = None
    game_context: str = ""  # 현재 게임 상황 컨텍스트
    
    def __post_init__(self):
        if self.active_characters is None:
            self.active_characters = []
        if self.session_log is None:
            self.session_log = []
        if self.turn_order is None:
            self.turn_order = []

# ===== 공유 게임 상태 =====
class GameStateManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.state = GameState()
        return cls._instance
    
    def update_context(self, new_context: str):
        self.state.game_context = new_context
        self.state.session_log.append(new_context)
    
    def get_context(self) -> str:
        return self.state.game_context
    
    def add_character(self, character: Character):
        self.state.active_characters.append(character)
    
    def get_character(self, name: str) -> Optional[Character]:
        for char in self.state.active_characters:
            if char.name.lower() == name.lower():
                return char
        return None

# 글로벌 게임 상태 매니저
game_state_manager = GameStateManager()

# ===== 도구들 =====
class DiceRollInput(BaseModel):
    sides: int = Field(default=20, description="주사위 면 수")
    count: int = Field(default=1, description="주사위 개수")
    modifier: int = Field(default=0, description="수정치")

class DiceRollTool(BaseTool):
    name: str = "roll_dice"
    description: str = "주사위를 굴립니다"
    args_schema: type[BaseModel] = DiceRollInput
    
    def _run(self, sides: int = 20, count: int = 1, modifier: int = 0) -> str:
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls) + modifier
        result = {
            "rolls": rolls,
            "modifier": modifier,
            "total": total,
            "description": f"{count}d{sides}+{modifier} = {rolls} + {modifier} = {total}"
        }
        return json.dumps(result, ensure_ascii=False)

class AbilityCheckInput(BaseModel):
    ability_score: int = Field(description="능력치 수치")
    difficulty: int = Field(default=10, description="난이도")

class AbilityCheckTool(BaseTool):
    name: str = "ability_check"
    description: str = "능력치 판정을 수행합니다"
    args_schema: type[BaseModel] = AbilityCheckInput
    
    def _run(self, ability_score: int, difficulty: int = 10) -> str:
        roll = random.randint(1, 20)
        modifier = (ability_score - 10) // 2
        total = roll + modifier
        success = total >= difficulty
        
        result = {
            "roll": roll,
            "modifier": modifier,
            "total": total,
            "difficulty": difficulty,
            "success": success,
            "description": f"d20({roll}) + 수정치({modifier}) = {total} vs DC{difficulty} - {'성공' if success else '실패'}"
        }
        return json.dumps(result, ensure_ascii=False)

class GameContextTool(BaseTool):
    name: str = "get_game_context"
    description: str = "현재 게임 상황과 컨텍스트를 가져옵니다"
    
    def _run(self) -> str:
        context = game_state_manager.get_context()
        characters = [asdict(char) for char in game_state_manager.state.active_characters]
        return json.dumps({
            "current_context": context,
            "characters": characters,
            "scene": game_state_manager.state.current_scene
        }, ensure_ascii=False)

class UpdateContextTool(BaseTool):
    name: str = "update_game_context"
    description: str = "게임 상황과 컨텍스트를 업데이트합니다"
    
    def _run(self, new_context: str) -> str:
        game_state_manager.update_context(new_context)
        return f"게임 컨텍스트가 업데이트되었습니다: {new_context}"

# Tool 인스턴스 생성
dice_tool = DiceRollTool()
ability_tool = AbilityCheckTool()
context_tool = GameContextTool()
update_context_tool = UpdateContextTool()

# ===== Agent 정의 =====

# Game Master Agent (실제 게임 진행)
game_master = Agent(
    role="게임 마스터",
    goal="플레이어 입력에 따라 즉시 반응하고 게임을 진행",
    backstory="""당신은 숙련된 D&D 게임 마스터입니다. 
    플레이어의 행동에 즉시 반응하고 흥미진진한 상황을 만들어냅니다.
    필요시 주사위를 굴리고 상황을 업데이트합니다.""",
    tools=[dice_tool, context_tool, update_context_tool],
    verbose=True,
    llm=f"openai/{MODEL_NAME}",
    max_tokens=1000,
    temperature=0.7
)

# Rules Advisor (필요시 조언)
rules_advisor = Agent(
    role="규칙 조언자",
    goal="복잡한 상황에서 D&D 규칙 조언 제공",
    backstory="""D&D 5판 규칙 전문가로서, 복잡한 상황에서만 
    규칙 해석과 판정 조언을 제공합니다.""",
    tools=[ability_tool, dice_tool],
    verbose=True,
    llm=f"openai/{MODEL_NAME}",
    max_tokens=800,
    temperature=0.3
)

# ===== 개선된 게임 엔진 =====
class ImprovedDnDGameEngine:
    def __init__(self):
        self.is_running = False
        self.offline_mode = False
        
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            response = litellm.completion(
                model=f"openai/{MODEL_NAME}",
                messages=[{"role": "user", "content": "테스트"}],
                api_base=API_BASE_URL,
                api_key=API_KEY,
                temperature=0.7,
                max_tokens=50,
                timeout=10
            )
            print("✅ LLM 연결 성공!")
            return True
        except Exception as e:
            print(f"❌ LLM 연결 실패: {e}")
            print("오프라인 모드로 전환합니다.")
            self.offline_mode = True
            return False
    
    def start_game(self) -> str:
        """게임 시작"""
        print("=== D&D Crew AI 게임 시작 ===")
        
        # 연결 테스트
        self.test_connection()
        
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
        
        game_state_manager.update_context(start_scenario)
        return start_scenario.strip()
    
    def process_input(self, player_input: str) -> str:
        """플레이어 입력 처리"""
        if not self.is_running:
            return "게임이 시작되지 않았습니다. start_game()을 먼저 호출하세요."
        
        print(f"\n🎲 플레이어: {player_input}")
        
        if self.offline_mode:
            return self._offline_response(player_input)
        
        try:
            # 단일 Task로 플레이어 입력 처리
            response_task = Task(
                description=f"""
                플레이어 액션: "{player_input}"
                
                현재 게임 상황을 파악하고 플레이어의 행동에 즉시 반응하세요.
                필요하면 주사위를 굴리고, 결과에 따라 새로운 상황을 묘사하세요.
                게임 컨텍스트를 업데이트하는 것도 잊지 마세요.
                
                응답 형식:
                - 행동 결과 묘사
                - 필요시 주사위 결과
                - 새로운 상황이나 선택지 제시
                """,
                agent=game_master,
                expected_output="플레이어 행동에 대한 즉각적인 반응과 새로운 상황"
            )
            
            # 단일 Agent로 즉시 처리
            single_crew = Crew(
                agents=[game_master],
                tasks=[response_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = single_crew.kickoff()
            return str(result)
            
        except Exception as e:
            print(f"❌ 처리 중 오류: {e}")
            return self._offline_response(player_input)
    
    def _offline_response(self, player_input: str) -> str:
        """오프라인 모드 응답"""
        responses = {
            "조사": "동굴을 조사하기로 했습니다. 어둠 속에서 이상한 소리가 들립니다...",
            "수락": "퀘스트를 수락했습니다! 동굴 입구로 향합니다.",
            "거절": "여관 주인이 실망한 표정을 짓습니다. 다른 모험가를 찾아보겠다고 합니다.",
            "인벤토리": "현재 소지품: 기본 검, 가죽 갑옷, 체력 물약 x2",
            "상태": "체력: 10/10, 컨디션: 양호"
        }
        
        for key, response in responses.items():
            if key in player_input.lower():
                return f"🎭 GM: {response}"
        
        # 기본 응답
        return f"🎭 GM: '{player_input}'에 대해 생각해봅니다... (오프라인 모드)"
    
    def get_help(self) -> str:
        """도움말"""
        return """
📖 **게임 명령어:**
- '조사하기' - 주변을 조사
- '인벤토리' - 소지품 확인  
- '상태' - 캐릭터 상태 확인
- '수락/거절' - 퀘스트 수락/거절
- '공격하기' - 전투 시작
- '도망가기' - 전투에서 도망
- 'save' - 게임 저장
- 'load' - 게임 불러오기
- 'quit' - 게임 종료
- 'help' - 이 도움말

자유롭게 행동을 입력하세요!
        """

# ===== 메인 실행 함수 =====
def run_game():
    """개선된 게임 실행"""
    try:
        game = ImprovedDnDGameEngine()
        
        # 게임 시작
        start_message = game.start_game()
        print(start_message)
        
        print("\n" + "="*50)
        print("게임이 시작되었습니다!")
        print("'help'를 입력하면 명령어를 볼 수 있습니다.")
        print("'quit'를 입력하면 게임을 종료합니다.")
        print("="*50)
        
        # 게임 루프
        while game.is_running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    print("🎮 게임을 종료합니다. 안녕히 가세요!")
                    break
                elif user_input.lower() == 'help':
                    print(game.get_help())
                elif user_input.lower() == 'save':
                    print("💾 게임 저장 기능은 추후 구현될 예정입니다.")
                elif user_input.lower() == 'load':
                    print("📁 게임 불러오기 기능은 추후 구현될 예정입니다.")
                else:
                    # 플레이어 입력 처리
                    response = game.process_input(user_input)
                    print(f"\n{response}")
                    
            except KeyboardInterrupt:
                print("\n\n🎮 게임을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                
    except Exception as e:
        print(f"❌ 게임 실행 중 치명적 오류: {e}")
        print("\n🔧 문제 해결:")
        print("1. VLLM 서버 상태 확인")
        print("2. 환경 변수 (.env) 확인")
        print("3. 의존성 패키지 재설치")

if __name__ == "__main__":
    run_game()