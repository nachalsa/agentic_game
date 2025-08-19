import logging
from datetime import datetime

from config import config
from models import Character, game_state_manager
from game_logic import DnDGameEngine

logger = logging.getLogger(__name__)

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
    print("\n🌟 주요 기능:")
    print("  • AI 게임 마스터와 실시간 상호작용")
    print("  • 자동 주사위 굴리기 및 능력치 판정")
    print("  • 게임 저장/불러오기 시스템")
    print("  • 상세한 캐릭터 상태 관리")
    print(f"  • LLM 모델: {config.MODEL_NAME}")
    print("\n게임을 시작하려면 Enter를 누르세요...")
    input()

def get_help() -> str:
    """도움말"""
    return f"""
📖 **D&D 게임 명령어:**

**기본 명령어:**
- 'help' - 이 도움말 표시
- 'quit' - 게임 종료
- 'save [파일명]' - 게임 저장
- 'load [파일명]' - 게임 불러오기
- 'saves' - 저장 파일 목록
- 'status' - 캐릭터 상태 확인

**게임 내 행동:**
- '조사하기' - 주변을 조사
- '인벤토리' - 소지품 확인  
- '수락/거절' - 퀘스트 수락/거절
- '공격하기' - 전투 시작
- '도망가기' - 전투에서 도망
- '말하기: [내용]' - NPC와 대화
- '마법 사용: [마법명]' - 마법 시전
- '아이템 사용: [아이템명]' - 아이템 사용

**시스템 정보:**
- 최대 입력 길이: {config.MAX_INPUT_LENGTH}자
- LLM 모델: {config.MODEL_NAME}
- API 서버: {config.API_BASE_URL}

자유롭게 행동을 입력하세요! AI 게임 마스터가 반응합니다.
    """

def handle_command(game: DnDGameEngine, user_input: str) -> bool:
    """명령어 처리 - 계속 진행할지 여부 반환"""
    
    if user_input.lower() == 'quit':
        print("🎮 게임을 종료합니다. 즐거운 모험이었습니다!")
        return False
    
    elif user_input.lower() == 'help':
        print(get_help())
    
    elif user_input.lower() == 'status':
        print(game.get_status())
    
    elif user_input.lower() == 'saves':
        print(game.list_saves())
    
    elif user_input.lower().startswith('save'):
        parts = user_input.split(' ', 1)
        filename = parts[1] if len(parts) > 1 else None
        print(game.save_game(filename))
    
    elif user_input.lower().startswith('load'):
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            print("❌ 파일명을 입력해주세요. 예: load save_20240101_120000.json")
            print(game.list_saves())
        else:
            filename = parts[1]
            print(game.load_game(filename))
    
    else:
        # 일반 게임 입력 처리
        print("🎭 AI 게임 마스터가 생각하는 중...")
        try:
            response = game.process_input(user_input)
            print(f"\n{response}")
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            print(f"\n❌ {str(e)}")
            print("잠시 후 다시 시도해주세요.")
    
    return True

def run_game():
    """게임 실행"""
    game = None
    
    try:
        print("🎲 D&D Crew AI 게임 엔진을 초기화하는 중...")
        
        # 게임 엔진 생성
        game = DnDGameEngine()
        
        print("🔗 LLM 서버 연결을 확인하는 중...")
        
        # 게임 시작 - 연결 테스트 포함
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
                
                # 명령어 처리
                should_continue = handle_command(game, user_input)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n🎮 게임을 종료합니다.")
                break
            except EOFError:
                print("\n\n🎮 입력 스트림이 종료되었습니다.")
                break
                
    except KeyboardInterrupt:
        print("\n🎮 게임 초기화가 중단되었습니다.")
    
    except (ConnectionError, TimeoutError) as e:
        print(f"\n❌ 연결 오류: {str(e)}")
        print("\n🔧 해결 방법:")
        print("1. LLM 서버가 실행 중인지 확인")
        print(f"2. 서버 주소 확인: {config.API_BASE_URL}")
        print("3. 네트워크 연결 상태 확인")
    
    except ValueError as e:
        print(f"\n❌ 설정 오류: {str(e)}")
        print("\n🔧 해결 방법:")
        print("1. .env 파일의 API 키 확인")
        print("2. LLM 모델명 확인")
        print("3. 환경 변수 설정 확인")
    
    except RuntimeError as e:
        print(f"\n❌ 실행 오류: {str(e)}")
        print("\n🔧 해결 방법:")
        print("1. 의존성 패키지 재설치: pip install -r requirements.txt")
        print("2. 로그 파일 확인: logs/ 디렉터리")
        print("3. 설정 파일 확인")
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        print(f"\n❌ 예상치 못한 오류가 발생했습니다.")
        print(f"오류 내용: {str(e)}")
        print("\n🔧 해결 방법:")
        print("1. 프로그램을 재시작해주세요")
        print("2. 로그 파일을 확인해주세요")
        print("3. 문제가 지속되면 설정을 확인해주세요")

def main():
    """메인 함수"""
    try:
        show_welcome()
        run_game()
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        print(f"❌ 프로그램을 시작할 수 없습니다: {str(e)}")
    finally:
        print("\n👋 D&D Crew AI를 이용해주셔서 감사합니다!")

if __name__ == "__main__":
    main()