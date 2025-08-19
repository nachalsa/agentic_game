import json
import random
import logging
import requests
import litellm
from typing import Dict, List, Any
from dataclasses import asdict

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from config import config
from models import game_state_manager, InputValidator

logger = logging.getLogger(__name__)

# ===== CrewAI 도구들 =====
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

# ===== 게임 엔진 =====
class DnDGameEngine:
    """D&D 게임 엔진 - 온라인 전용"""
    
    def __init__(self):
        self.is_running = False
        self._crew = None
        self._agents = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def test_connection(self) -> bool:
        """LLM 연결 테스트 - 필수"""
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
            raise ConnectionError(f"LLM 서버에 연결할 수 없습니다: {e}")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"❌ 시간 초과: {e}")
            raise TimeoutError(f"LLM 서버 응답 시간 초과: {e}")
        except litellm.AuthenticationError as e:
            self.logger.error(f"❌ 인증 오류: {e}")
            raise ValueError(f"API 키 인증 실패: {e}")
        except Exception as e:
            self.logger.error(f"❌ 예상치 못한 오류: {e}")
            raise RuntimeError(f"LLM 연결 중 오류 발생: {e}")
    
    def _get_crew(self):
        """Crew 인스턴스 가져오기"""
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
                raise RuntimeError(f"AI 에이전트 생성 실패: {e}")
        return self._crew
    
    def start_game(self) -> str:
        """게임 시작"""
        from models import Character  # 순환 import 방지
        
        self.logger.info("=== D&D Crew AI 게임 시작 ===")
        
        # 연결 테스트 (필수)
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
        
        game_state_manager.update_context(start_scenario.strip())
        return start_scenario.strip()
    
    def process_input(self, player_input: str) -> str:
        """플레이어 입력 처리"""
        if not self.is_running:
            raise RuntimeError("게임이 시작되지 않았습니다. start_game()을 먼저 호출하세요.")
        
        # 입력 검증
        sanitized_input = InputValidator.sanitize_input(player_input)
        if not sanitized_input:
            return "❌ 유효하지 않은 입력입니다."
        
        if not InputValidator.validate_command(sanitized_input):
            return "❌ 입력이 너무 깁니다. 간단하게 입력해주세요."
        
        self.logger.info(f"플레이어 입력: {sanitized_input}")
        
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
            
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"연결 오류: {e}")
            raise ConnectionError("LLM 서버와의 연결이 끊어졌습니다. 네트워크 상태를 확인해주세요.")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"시간 초과: {e}")
            raise TimeoutError("LLM 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
        except litellm.AuthenticationError as e:
            self.logger.error(f"인증 오류: {e}")
            raise ValueError("API 키 인증에 실패했습니다. 설정을 확인해주세요.")
        except Exception as e:
            self.logger.error(f"입력 처리 중 오류: {e}")
            raise RuntimeError(f"게임 처리 중 오류가 발생했습니다: {e}")
    
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
    
    def save_game(self, filename: str = None) -> str:
        """게임 저장"""
        try:
            if game_state_manager.save_game(filename):
                from datetime import datetime
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
                self.is_running = True  # 게임 로드 후 실행 상태로 변경
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