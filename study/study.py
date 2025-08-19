import os
import time
import logging
import requests
from datetime import datetime
from contextlib import contextmanager
import litellm
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'crewai_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 크루 AI 텔레메트리 비활성화
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# 환경 변수 가져오기 (또는 직접 설정)
MODEL_NAME = os.getenv("DEFAULT_LLM", "cpatonn/Devstral-Small-2507-AWQ")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://localhost:54321")
API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "300"))

# URL 정규화 - /v1이 이미 포함되어 있는지 확인
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """지수 백오프 재시도 로직"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            logger.warning(f"시도 {attempt + 1} 실패: {e}. {delay}초 후 재시도...")
            time.sleep(delay)

@contextmanager
def managed_session():
    """requests 세션 관리"""
    session = requests.Session()
    try:
        yield session
    finally:
        session.close()

def test_litellm_connection():
    """LiteLLM을 통한 연결 테스트"""
    def _test():
        logger.info("LiteLLM 연결 테스트 중...")
        
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": "user", "content": "안녕하세요, LiteLLM 테스트입니다."}],
            api_base=API_BASE_URL,
            api_key=API_KEY,
            temperature=0.7,
            max_tokens=100,
            timeout=TIMEOUT,
            drop_params=True  # 지원하지 않는 파라미터 자동 제거
        )
        
        test_content = response.choices[0].message.content
        logger.info(f"LiteLLM 테스트 성공: {test_content}")
        return True
    
    return retry_with_backoff(_test)

def setup_litellm_global_config():
    """LiteLLM 전역 설정 (CrewAI 호환성을 위해 필요)"""
    litellm.api_base = API_BASE_URL
    litellm.api_key = API_KEY
    litellm.drop_params = True
    litellm.set_verbose = True
    logger.info("LiteLLM 전역 설정 완료")

def create_agents():
    """에이전트 생성"""
    researcher = Agent(
        role='고급 연구 분석가',
        goal='최신 AI 트렌드에 대한 포괄적인 연구 수행',
        backstory='최신 AI 기술 및 트렌드를 밝히는 데 특화된 숙련된 연구원',
        verbose=True,
        allow_delegation=False,
        # CrewAI는 전역 LiteLLM 설정을 사용하므로 원래 방식 유지
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.7
    )

    writer = Agent(
        role='전문 콘텐츠 작가',
        goal='주어진 연구를 바탕으로 매력적인 블로그 게시물 작성',
        backstory='독자들에게 복잡한 개념을 명확하고 매력적으로 전달하는 데 뛰어난 작가',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.8
    )
    
    return researcher, writer

def create_tasks(researcher, writer):
    """작업 생성"""
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
    
    return research_task, write_task

def save_result(result):
    """결과를 파일로 저장"""
    if not result:
        logger.warning("저장할 결과가 없습니다.")
        return None
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ai_trends_blog_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# AI 트렌드 블로그 포스트\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(str(result))
        
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        return filename
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return None

def run_crew_with_error_handling():
    """에러 처리가 포함된 크루 실행 함수"""
    try:
        logger.info("설정 검증 완료")
        logger.info(f"모델: {MODEL_NAME}")
        logger.info(f"API Base: {API_BASE_URL}")
        logger.info(f"API Key: {'설정됨' if API_KEY else '설정되지 않음'}")
        logger.info(f"Timeout: {TIMEOUT}초")
        
        # 연결 테스트
        logger.info("=" * 50)
        logger.info("LiteLLM 연결 테스트 중...")
        test_litellm_connection()
        logger.info("✅ LiteLLM 연결 성공")
        
        # LiteLLM 전역 설정 (CrewAI 호환성을 위해 필요)
        setup_litellm_global_config()
        
        # 에이전트 및 작업 생성
        logger.info("에이전트 생성 중...")
        researcher, writer = create_agents()
        
        logger.info("작업 생성 중...")
        research_task, write_task = create_tasks(researcher, writer)
        
        # 크루 생성 및 실행
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True,
            max_execution_time=MAX_EXECUTION_TIME
        )
        
        logger.info("=" * 50)
        logger.info("크루 작업을 시작합니다... (1-2분 소요 예상)")
        logger.info("=" * 50)
        
        result = crew.kickoff()
        
        # 결과 저장
        saved_file = save_result(result)
        
        logger.info("\n" + "=" * 50)
        logger.info("## 크루 작업 완료 ##")
        logger.info("=" * 50 + "\n")
        print(result)  # 콘솔에도 출력
        
        if saved_file:
            logger.info(f"📁 결과가 '{saved_file}' 파일에도 저장되었습니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 크루 실행 중 오류 발생: {e}")
        logger.error(f"오류 타입: {type(e).__name__}")
        
        # 구체적인 문제 해결 방법 제시
        logger.info("\n🔧 문제 해결 방법:")
        logger.info("1. API 서버가 실행 중인지 확인하세요")
        logger.info("2. 모델 이름이 정확한지 확인하세요")
        logger.info("3. API 키와 엔드포인트가 올바른지 확인하세요")
        logger.info("4. 네트워크 연결을 확인하세요")
        logger.info("5. 타임아웃 설정을 늘려보세요")
        logger.info("6. 로그 파일을 확인하여 상세한 오류 정보를 확인하세요")
        
        return None

# 실행
if __name__ == "__main__":
    result = run_crew_with_error_handling()
    
    if result:
        print("\n✅ 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업이 실패했습니다. 로그 파일을 확인해보세요.")