import os
import litellm
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import random
import json
import datetime

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기 (새로운 모델로 업데이트)
MODEL_NAME = os.getenv("DEFAULT_LLM", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://localhost:54321")
API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")

# URL 정규화 - /v1이 이미 포함되어 있는지 확인
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

print(f"MODEL_NAME: {MODEL_NAME}")
print(f"API_BASE_URL: {API_BASE_URL}")
print(f"API_KEY: {'****' if API_KEY else 'None'}")

# LiteLLM 글로벌 설정
litellm.api_base = API_BASE_URL
litellm.api_key = API_KEY
litellm.drop_params = True  # 지원하지 않는 파라미터 자동 제거
litellm.set_verbose = True  # 디버깅을 위한 상세 로그

# D&D 게임 상수 정의
CHARACTER_CLASSES = {
    "전사": {"주요능력": "힘", "특징": "근접 전투 전문, 높은 체력", "무기": ["검", "도끼", "방패"]},
    "마법사": {"주요능력": "지능", "특징": "강력한 마법, 낮은 체력", "무기": ["지팡이", "완드", "두루마리"]},
    "도적": {"주요능력": "민첩", "특징": "은신과 함정 해제", "무기": ["단검", "활", "석궁"]},
    "성직자": {"주요능력": "지혜", "특징": "치유와 신성 마법", "무기": ["메이스", "홀리 심볼", "방패"]},
    "바드": {"주요능력": "매력", "특징": "다양한 기술과 버프", "무기": ["활", "검", "악기"]},
    "레인저": {"주요능력": "민첩", "특징": "자연 마법과 추적", "무기": ["활", "검", "창"]}
}

FANTASY_SETTINGS = {
    "중세 판타지": "전형적인 중세 유럽 판타지 세계, 성과 기사들",
    "다크 판타지": "어둡고 위험한 세계, 공포와 절망이 가득한 분위기",
    "하이 판타지": "마법이 강력하고 흔한 세계, 신화적 존재들",
    "도시 판타지": "현대 도시에 숨겨진 마법과 환상의 세계",
    "스팀펑크": "증기기관과 마법이 결합된 빅토리아 시대 분위기",
    "해적 판타지": "바다와 섬들의 모험, 해적들과 바다 괴물들"
}

def test_connection():
    """서버 연결을 테스트합니다."""
    try:
        import requests
        chat_url = API_BASE_URL + '/chat/completions'
        test_data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "연결 테스트입니다."}],
            "temperature": 0.15,
            "max_tokens": 50
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        response = requests.post(chat_url, json=test_data, headers=headers, timeout=30)
        print(f"서버 상태: {response.status_code}")
        
        if response.status_code == 200:
            print("서버 연결 성공!")
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                test_response = result['choices'][0]['message']['content']
                print(f"테스트 응답: {test_response}")
                return True
        else:
            print(f"서버 응답 오류: {response.text}")
            return False
    except Exception as e:
        print(f"서버 연결 실패: {e}")
        return False

def test_litellm_connection():
    """LiteLLM을 통한 연결 테스트"""
    try:
        print("LiteLLM 연결 테스트 중...")
        
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": "user", "content": "안녕하세요, LiteLLM 테스트입니다."}],
            api_base=API_BASE_URL,
            api_key=API_KEY,
            temperature=0.15,
            max_tokens=100,
            timeout=30
        )
        
        print(f"LiteLLM 테스트 성공: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"LiteLLM 연결 실패: {e}")
        print(f"오류 타입: {type(e).__name__}")
        return False

# 연결 테스트 실행
print("=" * 60)
print("🎲 D&D 게임 마스터 AI 시작")
print("=" * 60)
print("서버 연결 테스트 중...")

# 주석 처리: 연결 테스트 실패 시에도 계속 진행
# if not test_connection():
#     print("❌ 서버에 연결할 수 없습니다. 오프라인 모드로 진행합니다.")
# 
# print("\nLiteLLM 연결 테스트 중...")
# if not test_litellm_connection():
#     print("❌ LiteLLM 연결에 실패했습니다. 기본 설정으로 진행합니다.")

print("=" * 60)

# D&D 게임 에이전트 정의
dungeon_master = Agent(
    role='던전 마스터 (DM)',
    goal='흥미진진하고 몰입감 있는 D&D 모험을 진행하며 플레이어들에게 재미있는 경험 제공',
    backstory='''수십 년의 경험을 가진 베테랑 던전 마스터로, 창의적인 스토리텔링과 
    공정한 룰 적용으로 유명합니다. 플레이어들의 선택을 존중하며 예상치 못한 
    상황에도 유연하게 대응할 수 있습니다.''',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=3000,
    temperature=0.8  # 창의적인 스토리텔링을 위해 높은 온도
)

character_creator = Agent(
    role='캐릭터 생성 전문가',
    goal='플레이어의 요구사항에 맞는 독창적이고 균형잡힌 D&D 캐릭터 생성',
    backstory='''D&D 캐릭터 생성의 전문가로, 다양한 클래스와 종족의 조합을 
    통해 플레이어가 원하는 컨셉을 완벽하게 구현할 수 있습니다. 
    게임 밸런스와 롤플레이 재미를 모두 고려합니다.''',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2500,
    temperature=0.7
)

rules_advisor = Agent(
    role='룰 자문관',
    goal='D&D 5판 규칙을 정확하게 적용하고 플레이어들에게 명확한 안내 제공',
    backstory='''D&D 5판 규칙서를 완벽하게 숙지한 룰 전문가입니다. 
    복잡한 상황에서도 적절한 규칙을 적용하고, 플레이어들이 
    규칙을 이해할 수 있도록 친절하게 설명합니다.''',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2000,
    temperature=0.3  # 정확한 규칙 적용을 위해 낮은 온도
)

story_weaver = Agent(
    role='스토리 편직자',
    goal='플레이어들의 행동과 선택을 바탕으로 일관성 있고 흥미로운 내러티브 구성',
    backstory='''뛰어난 상상력과 스토리텔링 능력을 가진 작가로, 
    플레이어들의 예상치 못한 행동도 자연스럽게 스토리에 편입시켜 
    몰입감 있는 모험을 만들어냅니다.''',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=3000,
    temperature=0.9  # 매우 창의적인 스토리텔링
)

def get_game_setup():
    """게임 설정을 입력받습니다."""
    print("\n🎭 D&D 게임 설정을 시작합니다!")
    print("=" * 50)
    
    # 플레이어 수
    while True:
        try:
            num_players = input("\n플레이어 수를 입력하세요 (1-6, 기본값: 1): ").strip()
            if not num_players:
                num_players = 1
                break
            num_players = int(num_players)
            if 1 <= num_players <= 6:
                break
            else:
                print("1-6 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 게임 설정 선택
    print(f"\n📚 판타지 설정을 선택하세요:")
    settings = list(FANTASY_SETTINGS.keys())
    for i, setting in enumerate(settings, 1):
        print(f"{i}. {setting} - {FANTASY_SETTINGS[setting]}")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(settings)}, 기본값: 1): ").strip()
            if not choice:
                fantasy_setting = settings[0]
                break
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(settings):
                fantasy_setting = settings[choice_idx]
                break
            else:
                print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 모험 레벨
    while True:
        try:
            level = input("\n시작 레벨을 입력하세요 (1-20, 기본값: 1): ").strip()
            if not level:
                level = 1
                break
            level = int(level)
            if 1 <= level <= 20:
                break
            else:
                print("1-20 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 모험 길이
    campaign_length = input("\n모험 길이를 선택하세요 (단편/중편/장편, 기본값: 단편): ").strip() or "단편"
    
    return {
        'num_players': num_players,
        'fantasy_setting': fantasy_setting,
        'level': level,
        'campaign_length': campaign_length
    }

def create_character_task(player_num, game_setup):
    """캐릭터 생성 태스크를 만듭니다."""
    return Task(
        description=f'''플레이어 {player_num}를 위한 D&D 캐릭터를 생성하세요.

        게임 설정:
        - 판타지 설정: {game_setup['fantasy_setting']}
        - 시작 레벨: {game_setup['level']}
        - 모험 길이: {game_setup['campaign_length']}

        생성할 캐릭터 정보:
        1. 종족 및 클래스 추천 (설정에 맞는)
        2. 기본 능력치 (힘, 민첩, 체질, 지능, 지혜, 매력)
        3. 배경 스토리 (간단한 과거사)
        4. 성격 특성 및 목표
        5. 시작 장비 및 주문 (해당시)
        6. 특별한 기술이나 특성

        {CHARACTER_CLASSES}에서 적절한 클래스를 선택하고,
        플레이어가 쉽게 롤플레이할 수 있도록 구체적인 성격과 동기를 제공하세요.''',
        expected_output=f'플레이어 {player_num}를 위한 완전한 D&D 캐릭터 시트 (종족, 클래스, 능력치, 배경, 장비 포함)',
        agent=character_creator
    )

def create_campaign_task(game_setup, characters_context=None):
    """캠페인 생성 태스크를 만듭니다."""
    characters_info = ""
    if characters_context:
        characters_info = f"\n생성된 캐릭터들을 고려하여 모험을 설계하세요:\n{characters_context}"
    
    return Task(
        description=f'''"{game_setup['fantasy_setting']}" 설정에서 레벨 {game_setup['level']} 캐릭터들을 위한 
        {game_setup['campaign_length']} 모험을 설계하세요.

        모험 구성 요소:
        1. 메인 퀘스트와 목표
        2. 시작 지역과 주요 장소들
        3. 주요 NPC들 (적, 동료, 퀘스트 제공자)
        4. 예상되는 전투 및 도전과제들
        5. 보상과 전리품
        6. 사이드 퀘스트 아이디어들
        7. 스토리 훅과 반전 요소들

        난이도: {"초급자 친화적" if game_setup['level'] <= 3 else "중급자용" if game_setup['level'] <= 10 else "고급자용"}
        
        {characters_info}

        모험은 플레이어들이 {game_setup['num_players']}명이므로 그에 맞는 밸런스로 설계하세요.
        {FANTASY_SETTINGS[game_setup['fantasy_setting']]} 분위기를 잘 살려주세요.''',
        expected_output=f'{game_setup["campaign_length"]} {game_setup["fantasy_setting"]} 모험 시나리오 (퀘스트, 장소, NPC, 전투, 보상 포함)',
        agent=dungeon_master
    )

def create_session_guide_task(game_setup, campaign_context=None):
    """세션 가이드 생성 태스크를 만듭니다."""
    return Task(
        description=f'''생성된 캠페인을 바탕으로 첫 번째 게임 세션을 위한 상세한 DM 가이드를 작성하세요.

        세션 가이드 내용:
        1. 세션 개요 및 목표
        2. 오프닝 시나리오 (어떻게 시작할지)
        3. 주요 장면들과 이벤트 순서
        4. NPC 대화 스크립트 예시
        5. 전투 인카운터 상세 정보
        6. 가능한 플레이어 선택지들과 대응 방안
        7. 룰 참조사항 (자주 사용될 규칙들)
        8. 임기응변을 위한 팁들
        9. 세션 마무리 및 다음 세션 연결점

        대상: {"초보 DM" if game_setup['level'] <= 3 else "경험 있는 DM"}
        예상 플레이 시간: {"2-3시간" if game_setup['campaign_length'] == "단편" else "3-4시간"}

        실제 게임에서 바로 사용할 수 있도록 구체적이고 실용적으로 작성하세요.''',
        expected_output='첫 번째 D&D 세션을 위한 완전한 DM 가이드 (시나리오, 대화, 전투, 룰 참조 포함)',
        agent=story_weaver
    )

def run_dnd_game_setup():
    """D&D 게임 설정을 실행합니다."""
    try:
        # 게임 설정 입력받기
        game_setup = get_game_setup()
        
        print(f"\n🎯 게임 설정 완료:")
        print(f"   - 플레이어 수: {game_setup['num_players']}")
        print(f"   - 판타지 설정: {game_setup['fantasy_setting']}")
        print(f"   - 시작 레벨: {game_setup['level']}")
        print(f"   - 모험 길이: {game_setup['campaign_length']}")
        
        all_tasks = []
        all_results = {}
        
        # 1단계: 캐릭터 생성
        if input("\n🧙‍♂️ 캐릭터를 자동 생성하시겠습니까? (Y/n): ").lower() != 'n':
            print(f"\n{'='*60}")
            print(f"🧙‍♂️ {game_setup['num_players']}명의 캐릭터 생성 중...")
            print(f"{'='*60}")
            
            character_tasks = []
            for i in range(game_setup['num_players']):
                task = create_character_task(i+1, game_setup)
                character_tasks.append(task)
            
            # 캐릭터 생성 크루 실행
            character_crew = Crew(
                agents=[character_creator],
                tasks=character_tasks,
                process=Process.sequential,
                verbose=True,
                max_execution_time=300
            )
            
            character_results = character_crew.kickoff()
            all_results['characters'] = character_results
            
            print(f"\n✅ 캐릭터 생성 완료!")
            print(f"{'='*60}")
            print(character_results)
        
        # 2단계: 캠페인 생성
        print(f"\n{'='*60}")
        print(f"📖 캠페인 시나리오 생성 중...")
        print(f"{'='*60}")
        
        characters_context = all_results.get('characters', '')
        campaign_task = create_campaign_task(game_setup, characters_context)
        
        campaign_crew = Crew(
            agents=[dungeon_master, rules_advisor],
            tasks=[campaign_task],
            process=Process.sequential,
            verbose=True,
            max_execution_time=400
        )
        
        campaign_result = campaign_crew.kickoff()
        all_results['campaign'] = campaign_result
        
        print(f"\n✅ 캠페인 생성 완료!")
        print(f"{'='*60}")
        print(campaign_result)
        
        # 3단계: 세션 가이드 생성
        print(f"\n{'='*60}")
        print(f"📝 DM 세션 가이드 생성 중...")
        print(f"{'='*60}")
        
        session_task = create_session_guide_task(game_setup, campaign_result)
        
        session_crew = Crew(
            agents=[story_weaver, rules_advisor],
            tasks=[session_task],
            process=Process.sequential,
            verbose=True,
            max_execution_time=400
        )
        
        session_result = session_crew.kickoff()
        all_results['session_guide'] = session_result
        
        print(f"\n✅ 세션 가이드 생성 완료!")
        print(f"{'='*60}")
        print(session_result)
        
        # 결과 저장
        save_dnd_session(game_setup, all_results)
        
        print(f"\n🎉 D&D 게임 준비 완료!")
        print(f"📁 모든 자료가 파일로 저장되었습니다.")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ D&D 게임 설정 중 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        print("\n🔧 문제 해결 방법:")
        print("1. 서버가 실행 중인지 확인하세요")
        print("2. 네트워크 연결 상태를 확인하세요")
        print("3. 모델 설정이 올바른지 확인하세요")
        
        return None

def save_dnd_session(game_setup, results):
    """D&D 세션 결과를 파일로 저장합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        setting_name = "".join(c for c in game_setup['fantasy_setting'] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        filename = f"dnd_session_{setting_name}_lv{game_setup['level']}_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# D&D 게임 세션 자료\n\n")
            f.write(f"- **생성일시**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **플레이어 수**: {game_setup['num_players']}\n")
            f.write(f"- **판타지 설정**: {game_setup['fantasy_setting']}\n")
            f.write(f"- **시작 레벨**: {game_setup['level']}\n")
            f.write(f"- **모험 길이**: {game_setup['campaign_length']}\n\n")
            f.write("---\n\n")
            
            if 'characters' in results:
                f.write(f"## 🧙‍♂️ 플레이어 캐릭터들\n\n")
                f.write(str(results['characters']))
                f.write("\n\n---\n\n")
            
            if 'campaign' in results:
                f.write(f"## 📖 캠페인 시나리오\n\n")
                f.write(str(results['campaign']))
                f.write("\n\n---\n\n")
            
            if 'session_guide' in results:
                f.write(f"## 📝 DM 세션 가이드\n\n")
                f.write(str(results['session_guide']))
                f.write("\n\n---\n\n")
        
        print(f"📁 D&D 세션 자료가 '{filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"⚠️ 파일 저장 중 오류 발생: {e}")

def quick_encounter():
    """빠른 인카운터 생성"""
    print("\n⚔️ 빠른 인카운터 생성기")
    
    encounter_types = ["전투", "사회적", "탐험", "퍼즐", "혼합"]
    print("인카운터 유형을 선택하세요:")
    for i, enc_type in enumerate(encounter_types, 1):
        print(f"{i}. {enc_type}")
    
    choice = input(f"선택 (1-{len(encounter_types)}): ").strip()
    try:
        encounter_type = encounter_types[int(choice) - 1]
    except:
        encounter_type = "전투"
    
    level = input("파티 레벨 (1-20, 기본값: 3): ").strip() or "3"
    
    encounter_task = Task(
        description=f'''레벨 {level} 파티를 위한 {encounter_type} 인카운터를 생성하세요.

        포함 사항:
        1. 인카운터 설정 및 배경
        2. 목표 및 성공 조건
        3. 적 또는 장애물 정보
        4. 필요한 주사위 굴림 및 DC
        5. 가능한 결과들
        6. 보상 및 후속 연결점

        밸런스가 잘 맞고 재미있는 인카운터로 만들어 주세요.''',
        expected_output=f'레벨 {level} 파티용 {encounter_type} 인카운터 (설정, 룰, 보상 포함)',
        agent=dungeon_master
    )
    
    encounter_crew = Crew(
        agents=[dungeon_master, rules_advisor],
        tasks=[encounter_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = encounter_crew.kickoff()
    print(f"\n✅ {encounter_type} 인카운터 생성 완료!")
    print("=" * 60)
    print(result)
    
    return result

# 메인 실행부
if __name__ == "__main__":
    print("\n🎲 D&D 게임 마스터 AI")
    print("=" * 60)
    
    while True:
        print("\n모드를 선택하세요:")
        print("1. 🎭 새 D&D 게임 설정")
        print("2. ⚔️ 빠른 인카운터 생성")
        print("3. 📚 캐릭터 클래스 정보")
        print("4. 🌍 판타지 설정 보기")
        print("5. 🚪 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == "1":
            result = run_dnd_game_setup()
            if result:
                print("\n✅ D&D 게임 설정이 성공적으로 완료되었습니다!")
            
        elif choice == "2":
            quick_encounter()
            
        elif choice == "3":
            print("\n📚 사용 가능한 캐릭터 클래스:")
            print("=" * 50)
            for cls, info in CHARACTER_CLASSES.items():
                print(f"\n🗡️ {cls}")
                print(f"   주요능력: {info['주요능력']}")
                print(f"   특징: {info['특징']}")
                print(f"   무기: {', '.join(info['무기'])}")
            input("\n계속하려면 Enter를 누르세요...")
            
        elif choice == "4":
            print("\n🌍 판타지 설정들:")
            print("=" * 50)
            for setting, desc in FANTASY_SETTINGS.items():
                print(f"\n🏰 {setting}")
                print(f"   {desc}")
            input("\n계속하려면 Enter를 누르세요...")
            
        elif choice == "5":
            print("\n🎲 즐거운 모험 되세요! 메이 유어 다이스 롤 하이!")
            break
            
        else:
            print("❌ 올바른 번호를 선택하세요.")
