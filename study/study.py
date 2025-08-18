import os
import litellm
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

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

# 1. 에이전트 정의 (LiteLLM 직접 사용)
researcher = Agent(
    role='고급 연구 분석가',
    goal='최신 AI 트렌드에 대한 포괄적인 연구 수행',
    backstory='최신 AI 기술 및 트렌드를 밝히는 데 특화된 숙련된 연구원',
    verbose=True,
    allow_delegation=False,
    # LiteLLM 사용을 위한 설정
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2000,
    temperature=0.7
)

writer = Agent(
    role='전문 콘텐츠 작가',
    goal='주어진 연구를 바탕으로 매력적인 블로그 게시물 작성',
    backstory='독자들에게 복잡한 개념을 명확하고 매력적으로 전달하는 데 뛰어난 작가',
    verbose=True,
    allow_delegation=False,
    # 같은 LiteLLM 설정 사용
    llm=f"openai/{MODEL_NAME}",
    api_base=API_BASE_URL,
    api_key=API_KEY,
    max_tokens=2000,
    temperature=0.8
)

# 2. 작업 정의
research_task = Task(
    description='''2025년 최신 AI 트렌드를 조사하고 주요 발견 사항을 요약합니다. 
    다음 영역에 중점을 둡니다:
    1. 생성형 AI의 최신 발전사항
    2. 대규모 언어 모델(LLM)의 새로운 기능
    3. AI 윤리 및 규제 동향
    4. 산업별 AI 적용 사례
    
    각 영역에 대해 구체적인 예시와 통계를 포함하세요.''',
    expected_output='2025년 AI 트렌드에 대한 주요 통찰력, 통계 및 실제 예시를 포함하는 300단어 분량의 자세한 연구 요약',
    agent=researcher
)

write_task = Task(
    description='''연구 요약을 바탕으로 "2025년 AI 트렌드: 미래를 바꾸는 기술들"이라는 제목의 블로그 게시물을 작성합니다.
    
    요구사항:
    - 500-700단어 분량
    - 매력적인 도입부와 결론
    - 명확하고 이해하기 쉬운 언어 사용
    - 독자의 관심을 끄는 부제목 활용
    - 실제 사례나 예시 포함
    
    대상 독자: AI에 관심있는 일반 대중''',
    expected_output='독자의 참여를 유도하고 정보가 풍부하며 잘 구성된 500-700단어 분량의 블로그 게시물',
    agent=writer,
    context=[research_task]
)

# 3. 크루 정의 및 실행
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
    max_execution_time=600  # 최대 실행 시간 10분
)

def run_crew_with_error_handling():
    """에러 처리가 포함된 크루 실행 함수"""
    try:
        print("=" * 50)
        print("크루 작업을 시작합니다...")
        print("=" * 50)
        
        result = crew.kickoff()
        
        print("\n" + "=" * 50)
        print("## 크루 작업 완료 ##")
        print("=" * 50 + "\n")
        print(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 크루 실행 중 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        # 일반적인 해결 방법 제시
        print("\n🔧 문제 해결 방법:")
        print("1. VLLM 서버가 실행 중인지 확인하세요")
        print("2. 모델 이름이 정확한지 확인하세요")
        print("3. API 키와 엔드포인트가 올바른지 확인하세요")
        print("4. LiteLLM 버전을 확인하세요: pip install --upgrade litellm")
        
        # 상세 디버깅 정보
        print(f"\n🔍 디버깅 정보:")
        print(f"모델: {MODEL_NAME}")
        print(f"API Base: {API_BASE_URL}")
        print(f"API Key: {'설정됨' if API_KEY else '설정되지 않음'}")
        
        return None

# 실행
if __name__ == "__main__":
    result = run_crew_with_error_handling()
    
    if result:
        print("\n✅ 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업이 실패했습니다.")