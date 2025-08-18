import os
import litellm
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import random
import datetime

# .env 파일 로드
load_dotenv()

# 크루 AI 텔레메트리 비활성화
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# 환경 변수 가져오기 (또는 직접 설정)
MODEL_NAME = os.getenv("DEFAULT_LLM", "cpatonn/Devstral-Small-2507-AWQ")
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

# 난이도별 알고리즘 문제 카테고리
ALGORITHM_CATEGORIES = {
    "초급": {
        "배열과 문자열": ["배열 순회", "문자열 조작", "간단한 정렬", "선형 탐색"],
        "기본 수학": ["피보나치", "소수 판별", "최대공약수", "팩토리얼"],
        "기본 자료구조": ["스택 구현", "큐 구현", "리스트 조작"],
        "간단한 정렬": ["버블 정렬", "선택 정렬", "삽입 정렬"]
    },
    "중급": {
        "고급 정렬": ["퀵 정렬", "병합 정렬", "힙 정렬"],
        "탐색 알고리즘": ["이진 탐색", "깊이우선탐색", "너비우선탐색"],
        "동적 계획법": ["동전 문제", "배낭 문제", "최장증가수열"],
        "그리디 알고리즘": ["활동 선택", "최소 신장 트리", "허프만 코딩"],
        "해시와 맵": ["해시 테이블", "두 포인터", "슬라이딩 윈도우"]
    },
    "고급": {
        "그래프 알고리즘": ["다익스트라", "플로이드-워셜", "크루스칼", "프림"],
        "고급 동적계획법": ["비트마스크 DP", "트리 DP", "구간 DP"],
        "문자열 알고리즘": ["KMP", "라빈-카프", "트라이", "접미사 배열"],
        "고급 자료구조": ["세그먼트 트리", "펜윅 트리", "유니온 파인드"],
        "백트래킹": ["N-Queens", "스도쿠", "조합 생성"]
    }
}

def test_vllm_connection():
    """VLLM 서버 연결을 테스트합니다."""
    try:
        import requests
        # VLLM 서버의 chat/completions 엔드포인트 테스트
        chat_url = API_BASE_URL + '/chat/completions'
        test_data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "연결 테스트입니다."}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        response = requests.post(chat_url, json=test_data, headers=headers, timeout=30)
        print(f"VLLM 서버 상태: {response.status_code}")
        
        if response.status_code == 200:
            print("VLLM 서버 연결 성공!")
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                test_response = result['choices'][0]['message']['content']
                print(f"테스트 응답: {test_response}")
                return True
        else:
            print(f"VLLM 서버 응답 오류: {response.text}")
            return False
    except Exception as e:
        print(f"VLLM 서버 연결 실패: {e}")
        return False

def test_litellm_connection():
    """LiteLLM을 통한 연결 테스트"""
    try:
        print("LiteLLM 연결 테스트 중...")
        
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",  # OpenAI 호환 형식으로 지정
            messages=[{"role": "user", "content": "안녕하세요, LiteLLM 테스트입니다."}],
            api_base=API_BASE_URL,
            api_key=API_KEY,
            temperature=0.7,
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
print("=" * 50)
print("VLLM 서버 연결 테스트 중...")
if not test_vllm_connection():
    print("❌ VLLM 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    exit(1)

print("\nLiteLLM 연결 테스트 중...")
if not test_litellm_connection():
    print("❌ LiteLLM 연결에 실패했습니다.")
    exit(1)

print("=" * 50)

# 1. 에이전트 정의
problem_creator = Agent(
    role='알고리즘 문제 출제자',
    goal='난이도에 맞는 창의적이고 교육적인 알고리즘 문제 생성',
    backstory='다양한 난이도의 알고리즘 문제를 출제하는 전문가로, 학습자의 실력 향상에 도움이 되는 문제를 만드는 데 특화됨',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2000,
    temperature=0.8
)

solution_provider = Agent(
    role='솔루션 제공자',
    goal='문제에 대한 최적의 해답과 다양한 접근 방법 제시',
    backstory='알고리즘 문제를 다양한 방법으로 해결하고, 각 방법의 장단점을 분석하는 전문가',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2500,
    temperature=0.6
)

tutor = Agent(
    role='알고리즘 튜터',
    goal='문제와 해답에 대한 상세한 해설과 학습 가이드 제공',
    backstory='학습자가 문제를 이해하고 유사한 문제를 혼자 풀 수 있도록 돕는 친절한 튜터',
    verbose=True,
    allow_delegation=False,
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2500,
    temperature=0.7
)

def display_categories():
    """사용 가능한 카테고리를 표시합니다."""
    print("\n📚 사용 가능한 알고리즘 카테고리:")
    print("=" * 50)
    
    for difficulty, categories in ALGORITHM_CATEGORIES.items():
        print(f"\n🎯 {difficulty}:")
        for i, (category, topics) in enumerate(categories.items(), 1):
            print(f"  {i}. {category}")
            print(f"     └ 포함 주제: {', '.join(topics[:3])}{'...' if len(topics) > 3 else ''}")

def get_user_preferences():
    """사용자의 학습 선호도를 입력받습니다."""
    print("\n🎓 알고리즘 문제 출제 AI에 오신 것을 환영합니다!")
    print("=" * 50)
    
    display_categories()
    
    # 난이도 선택
    print(f"\n난이도를 선택하세요:")
    difficulties = list(ALGORITHM_CATEGORIES.keys())
    for i, diff in enumerate(difficulties, 1):
        print(f"{i}. {diff}")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(difficulties)}): ").strip()
            if not choice:
                difficulty = "초급"  # 기본값
                break
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(difficulties):
                difficulty = difficulties[choice_idx]
                break
            else:
                print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 카테고리 선택
    categories = list(ALGORITHM_CATEGORIES[difficulty].keys())
    print(f"\n{difficulty} 난이도의 카테고리를 선택하세요:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    print(f"{len(categories) + 1}. 랜덤 선택")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(categories) + 1}): ").strip()
            if not choice:
                category = random.choice(categories)  # 기본값
                break
            choice_idx = int(choice) - 1
            if choice_idx == len(categories):  # 랜덤 선택
                category = random.choice(categories)
                break
            elif 0 <= choice_idx < len(categories):
                category = categories[choice_idx]
                break
            else:
                print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 문제 수 선택
    while True:
        try:
            num_problems = input("\n출제할 문제 수를 입력하세요 (1-5, 기본값: 1): ").strip()
            if not num_problems:
                num_problems = 1
                break
            num_problems = int(num_problems)
            if 1 <= num_problems <= 5:
                break
            else:
                print("1-5 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    return difficulty, category, num_problems

def create_problem_tasks(difficulty, category, problem_number):
    """문제 생성을 위한 태스크들을 생성합니다."""
    
    topics = ALGORITHM_CATEGORIES[difficulty][category]
    selected_topic = random.choice(topics)
    
    # 문제 출제 태스크
    problem_task = Task(
        description=f'''난이도 {difficulty}, 카테고리 '{category}', 주제 '{selected_topic}'에 해당하는 알고리즘 문제를 출제하세요.

        문제 요구사항:
        1. 명확하고 구체적인 문제 설명
        2. 입력과 출력 형식 명시
        3. 제약 조건 설정
        4. 2-3개의 예제 입출력
        5. 실제 코딩 테스트에서 나올 법한 현실적인 문제
        
        {difficulty} 수준에 맞는 적절한 복잡도로 출제하세요.
        문제는 창의적이면서도 해당 알고리즘 개념을 잘 활용할 수 있어야 합니다.''',
        expected_output=f'{selected_topic} 관련 {difficulty} 난이도의 완전한 알고리즘 문제 (설명, 입출력, 제약조건, 예제 포함)',
        agent=problem_creator
    )
    
    # 솔루션 제공 태스크
    solution_task = Task(
        description=f'''출제된 문제에 대한 완전한 해답을 제공하세요.

        포함 사항:
        1. 문제 분석과 접근 방법
        2. 최적의 알고리즘 설명
        3. Python으로 구현된 완전한 코드
        4. 시간 복잡도와 공간 복잡도 분석
        5. 가능한 다른 접근 방법 (있다면)
        6. 코드 실행 결과 시뮬레이션
        
        코드는 깔끔하고 주석이 잘 달려 있어야 합니다.''',
        expected_output=f'문제에 대한 완전한 해답 (분석, 알고리즘, 구현 코드, 복잡도 분석 포함)',
        agent=solution_provider,
        context=[problem_task]
    )
    
    # 튜터링 태스크
    tutorial_task = Task(
        description=f'''문제와 해답을 바탕으로 상세한 학습 가이드를 작성하세요.

        가이드 내용:
        1. 문제에서 요구하는 핵심 개념 설명
        2. 해결 과정의 단계별 설명
        3. 코드의 각 부분에 대한 상세 해설
        4. 자주 하는 실수와 주의사항
        5. 비슷한 유형의 문제들
        6. 추가 연습 문제 제안 (2-3개)
        
        초보자도 이해할 수 있도록 친근하고 상세하게 설명하세요.''',
        expected_output=f'문제와 해답에 대한 완전한 학습 가이드 (개념 설명, 단계별 해설, 연습 문제 포함)',
        agent=tutor,
        context=[problem_task, solution_task]
    )
    
    return [problem_task, solution_task, tutorial_task], selected_topic

def run_problem_session():
    """문제 출제 세션을 실행합니다."""
    try:
        # 사용자 선호도 입력받기
        difficulty, category, num_problems = get_user_preferences()
        
        print(f"\n🎯 설정된 조건:")
        print(f"   - 난이도: {difficulty}")
        print(f"   - 카테고리: {category}")
        print(f"   - 문제 수: {num_problems}")
        
        all_results = []
        
        for i in range(num_problems):
            print(f"\n{'='*60}")
            print(f"📝 문제 {i+1}/{num_problems} 생성 중...")
            print(f"{'='*60}")
            
            # 각 문제별 태스크 생성
            tasks, topic = create_problem_tasks(difficulty, category, i+1)
            
            # 크루 생성 및 실행
            crew = Crew(
                agents=[problem_creator, solution_provider, tutor],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                max_execution_time=600
            )
            
            print(f"🔍 주제: {topic}")
            result = crew.kickoff()
            
            problem_result = {
                'number': i + 1,
                'topic': topic,
                'difficulty': difficulty,
                'category': category,
                'content': result
            }
            
            all_results.append(problem_result)
            
            # 개별 문제 결과 출력
            print(f"\n{'='*60}")
            print(f"✅ 문제 {i+1} 완료!")
            print(f"{'='*60}")
            print(result)
            
            # 진행 상황 표시
            if i < num_problems - 1:
                print(f"\n⏳ 다음 문제 준비 중... ({i+2}/{num_problems})")
        
        # 전체 세션 결과 저장
        save_session_results(difficulty, category, all_results)
        
        print(f"\n🎉 총 {num_problems}개 문제 생성 완료!")
        print(f"📁 결과가 파일로 저장되었습니다.")
        
        # 추가 세션 제안
        if input("\n🔄 다른 문제를 더 풀어보시겠습니까? (y/N): ").lower() == 'y':
            run_problem_session()
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ 문제 생성 중 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        print("\n🔧 문제 해결 방법:")
        print("1. VLLM 서버가 실행 중인지 확인하세요")
        print("2. 네트워크 연결 상태를 확인하세요")
        print("3. 모델 설정이 올바른지 확인하세요")
        
        return None

def save_session_results(difficulty, category, results):
    """세션 결과를 파일로 저장합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_category = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"algorithm_problems_{difficulty}_{safe_category}_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 알고리즘 문제 세션 결과\n\n")
            f.write(f"- **생성일시**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **난이도**: {difficulty}\n")
            f.write(f"- **카테고리**: {category}\n")
            f.write(f"- **총 문제 수**: {len(results)}\n\n")
            f.write("---\n\n")
            
            for result in results:
                f.write(f"## 문제 {result['number']} - {result['topic']}\n\n")
                f.write(f"**주제**: {result['topic']}\n")
                f.write(f"**난이도**: {result['difficulty']}\n")
                f.write(f"**카테고리**: {result['category']}\n\n")
                f.write(str(result['content']))
                f.write("\n\n---\n\n")
        
        print(f"📁 세션 결과가 '{filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"⚠️ 파일 저장 중 오류 발생: {e}")

def practice_mode():
    """연습 모드: 특정 약점을 집중 연습"""
    print("\n🎯 연습 모드: 특정 알고리즘을 집중 연습합니다!")
    
    # 약점 진단
    print("\n어떤 알고리즘 유형이 어려우신가요?")
    weak_areas = []
    
    for difficulty, categories in ALGORITHM_CATEGORIES.items():
        print(f"\n{difficulty} 난이도:")
        for i, category in enumerate(categories.keys(), 1):
            print(f"  {i}. {category}")
    
    selected = input("\n어려운 카테고리를 입력하세요 (예: 동적 계획법): ").strip()
    
    # 맞춤형 문제 생성
    found_category = None
    found_difficulty = None
    
    for difficulty, categories in ALGORITHM_CATEGORIES.items():
        if selected in categories:
            found_category = selected
            found_difficulty = difficulty
            break
    
    if found_category:
        print(f"\n🎯 '{found_category}' ({found_difficulty}) 집중 연습을 시작합니다!")
        
        # 연습용 문제 5개 생성
        tasks, topic = create_problem_tasks(found_difficulty, found_category, 1)
        crew = Crew(
            agents=[problem_creator, solution_provider, tutor],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        print(result)
        
    else:
        print("❌ 해당 카테고리를 찾을 수 없습니다. 기본 모드로 전환합니다.")
        run_problem_session()

# 메인 실행부
if __name__ == "__main__":
    print("\n🚀 알고리즘 문제 출제 및 해설 AI")
    print("=" * 50)
    
    while True:
        print("\n모드를 선택하세요:")
        print("1. 📝 일반 문제 출제 모드")
        print("2. 🎯 연습 모드 (약점 집중)")
        print("3. 📊 카테고리 보기")
        print("4. 🚪 종료")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            result = run_problem_session()
            if result:
                print("\n✅ 문제 세션이 성공적으로 완료되었습니다!")
            
        elif choice == "2":
            practice_mode()
            
        elif choice == "3":
            display_categories()
            input("\n계속하려면 Enter를 누르세요...")
            
        elif choice == "4":
            print("\n👋 알고리즘 학습을 응원합니다! 수고하셨습니다!")
            break
            
        else:
            print("❌ 올바른 번호를 선택하세요.")