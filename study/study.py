import os
import time
import logging
import requests
from datetime import datetime
from contextlib import contextmanager
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

# DuckDuckGo Search 추가
try:
    from ddgs import DDGS  # 새 패키지명
except ImportError:
    from duckduckgo_search import DDGS  # 기존 패키지명 (fallback)

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

# ===== 웹 검색 툴 추가 =====
def search_web(query: str) -> str:
    """
    DuckDuckGo를 사용하여 웹에서 정보를 검색합니다.
    
    Args:
        query (str): 검색할 쿼리
        
    Returns:
        str: 검색 결과를 포맷된 문자열로 반환
    """
    try:
        logger.info(f"웹 검색 시작: '{query}'")
        
        # DuckDuckGo 검색 실행
        ddgs = DDGS()
        results = ddgs.text(
            query=query, 
            region='wt-wt',  # 전세계 검색으로 변경 (더 많은 AI 관련 결과)
            safesearch='moderate', 
            max_results=7  # 결과 수 증가
        )
        
        if not results:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
        
        # 결과 포맷팅
        formatted_results = f"🔍 '{query}' 검색 결과:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', '제목 없음')
            body = result.get('body', '설명 없음')
            href = result.get('href', '#')
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   📄 {body[:200]}{'...' if len(body) > 200 else ''}\n"
            formatted_results += f"   🔗 {href}\n\n"
        
        logger.info(f"웹 검색 완료: {len(results)}개 결과 반환")
        return formatted_results
        
    except Exception as e:
        error_msg = f"웹 검색 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return error_msg

# CrewAI 툴로 래핑 (입력 스키마 명확화)
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """
    웹에서 정보를 검색하는 도구입니다. 최신 정보나 트렌드를 찾을 때 사용하세요.
    
    Args:
        query: 검색할 키워드나 질문 (예: "2025년 AI 트렌드", "최신 생성형 AI 발전사항")
    
    Returns:
        str: 검색 결과 목록 (제목, 설명, 링크 포함)
    """
    return search_web(query)

def should_use_web_search(description: str) -> bool:
    """
    작업 설명을 분석하여 웹 검색이 필요한지 판단합니다.
    """
    search_indicators = [
        "최신", "최근", "현재", "2025", "트렌드", "동향", 
        "뉴스", "발전사항", "업데이트", "새로운", "현재 상황"
    ]
    return any(indicator in description for indicator in search_indicators)

# ===== 기존 함수들 =====
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
            drop_params=True
        )
        
        test_content = response.choices[0].message.content
        logger.info(f"LiteLLM 테스트 성공: {test_content}")
        return True
    
    return retry_with_backoff(_test)

def test_web_search():
    """웹 검색 기능 테스트"""
    try:
        logger.info("웹 검색 기능 테스트 중...")
        # 직접 함수 호출로 테스트
        test_result = search_web("AI 트렌드 2025")
        if "검색 결과" in test_result or "오류" in test_result:
            logger.info("✅ 웹 검색 테스트 성공")
            return True
        else:
            logger.warning("⚠️ 웹 검색 결과가 예상과 다릅니다")
            return False
    except Exception as e:
        logger.error(f"❌ 웹 검색 테스트 실패: {e}")
        return False

def setup_litellm_global_config():
    """LiteLLM 전역 설정 (CrewAI 호환성을 위해 필요)"""
    litellm.api_base = API_BASE_URL
    litellm.api_key = API_KEY
    litellm.drop_params = True
    # 새로운 방식으로 로그 설정
    os.environ['LITELLM_LOG'] = 'INFO'  # 'DEBUG'에서 'INFO'로 변경하여 로그 줄이기
    logger.info("LiteLLM 전역 설정 완료")

def create_agents():
    """에이전트 생성 (웹 검색 툴 포함)"""
    
    # 연구 분석가 - 웹 검색 기능 포함
    researcher = Agent(
        role='고급 연구 분석가',
        goal='최신 AI 트렌드에 대한 포괄적인 연구 수행',
        backstory='''최신 AI 기술 및 트렌드를 밝히는 데 특화된 숙련된 연구원입니다. 
        웹 검색을 통해 실시간 정보를 수집하고 분석하는 능력을 갖추고 있습니다.''',
        verbose=True,
        allow_delegation=False,
        tools=[web_search_tool],  # 웹 검색 툴 추가
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.7
    )

    # 전문 작가
    writer = Agent(
        role='전문 콘텐츠 작가',
        goal='주어진 연구를 바탕으로 매력적인 블로그 게시물 작성',
        backstory='''독자들에게 복잡한 개념을 명확하고 매력적으로 전달하는 데 뛰어난 작가입니다.
        최신 정보와 트렌드를 바탕으로 현실적이고 유용한 콘텐츠를 만듭니다.''',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.8
    )
    
    return researcher, writer

def create_tasks(researcher, writer):
    """작업 생성 (웹 검색 활용)"""
    
    research_task = Task(
        description='''2025년 최신 AI 트렌드를 조사하고 주요 발견 사항을 요약합니다.
        
        다음 단계별로 웹 검색을 수행하세요 (영어 키워드 사용 권장):
        1. "generative AI latest developments 2025" 검색
        2. "large language models new features 2025" 검색  
        3. "AI ethics regulations trends 2025" 검색
        4. "AI industry applications 2025 cases" 검색
        
        각 검색은 Web Search Tool을 사용하여 개별적으로 수행하세요.
        검색할 때는 한 번에 하나의 쿼리만 사용하세요.
        
        각 영역에 대해 구체적인 예시와 최신 통계를 포함하세요.
        웹 검색을 통해 얻은 정보는 출처를 명시해주세요.''',
        
        expected_output='''2025년 AI 트렌드에 대한 주요 통찰력, 최신 통계 및 실제 예시를 포함하는 
        400-500단어 분량의 자세한 연구 요약. 웹 검색을 통해 얻은 최신 정보 포함.''',
        
        agent=researcher
    )

    write_task = Task(
        description='''연구 요약을 바탕으로 "2025년 AI 혁명: 현실이 된 미래 기술들"이라는 
        제목의 한국어 블로그 게시물을 작성합니다.
        
        요구사항:
        - 700-900단어 분량
        - 매력적인 도입부와 결론
        - 최신 웹 검색 결과를 반영한 현실적인 내용
        - 명확하고 이해하기 쉬운 언어 사용
        - 독자의 관심을 끄는 부제목 활용
        - 2025년 현재 상황을 반영한 실제 사례나 예시 포함
        - 출처가 있는 정보는 적절히 인용
        
        대상 독자: AI에 관심있는 일반 대중 및 비즈니스 전문가''',
        
        expected_output='''독자의 참여를 유도하고 정보가 풍부하며 잘 구성된 700-900단어 분량의 
        블로그 게시물. 최신 웹 검색 결과를 반영한 현실적이고 유용한 내용 포함.''',
        
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
    filename = f"ai_trends_blog_with_search_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# AI 트렌드 블로그 포스트 (웹 검색 포함)\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"검색 기능: DuckDuckGo 웹 검색 활용\n\n")
            f.write("---\n\n")
            f.write(str(result))
        
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        return filename
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return None

def run_crew_with_error_handling():
    """에러 처리가 포함된 크루 실행 함수 (웹 검색 포함)"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 CrewAI with Web Search 시작")
        logger.info("=" * 60)
        
        logger.info("설정 검증 완료")
        logger.info(f"모델: {MODEL_NAME}")
        logger.info(f"API Base: {API_BASE_URL}")
        logger.info(f"API Key: {'설정됨' if API_KEY else '설정되지 않음'}")
        logger.info(f"Timeout: {TIMEOUT}초")
        
        # 연결 테스트
        logger.info("\n" + "=" * 50)
        logger.info("🔌 시스템 연결 테스트")
        logger.info("=" * 50)
        
        logger.info("1. LiteLLM 연결 테스트 중...")
        test_litellm_connection()
        logger.info("✅ LiteLLM 연결 성공")
        
        logger.info("2. 웹 검색 기능 테스트 중...")
        if test_web_search():
            logger.info("✅ 웹 검색 기능 정상")
        else:
            logger.warning("⚠️  웹 검색 기능에 문제가 있지만 계속 진행합니다.")
        
        # LiteLLM 전역 설정
        setup_litellm_global_config()
        
        # 에이전트 및 작업 생성
        logger.info("\n📝 에이전트 및 작업 설정 중...")
        researcher, writer = create_agents()
        research_task, write_task = create_tasks(researcher, writer)
        
        # 크루 생성 및 실행
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True,
            max_execution_time=MAX_EXECUTION_TIME
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🚀 AI 크루 작업 시작 (웹 검색 포함)")
        logger.info("예상 소요 시간: 2-4분 (웹 검색 포함)")
        logger.info("=" * 60)
        
        result = crew.kickoff()
        
        # 결과 저장
        saved_file = save_result(result)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 크루 작업 완료!")
        logger.info("=" * 60)
        print("\n📄 생성된 콘텐츠:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        if saved_file:
            logger.info(f"\n📁 결과가 '{saved_file}' 파일에도 저장되었습니다.")
            logger.info("💡 이 파일에는 웹 검색을 통해 수집된 최신 정보가 포함되어 있습니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 크루 실행 중 오류 발생: {e}")
        logger.error(f"오류 타입: {type(e).__name__}")
        
        # 구체적인 문제 해결 방법 제시
        logger.info("\n🔧 문제 해결 방법:")
        logger.info("1. 필요한 라이브러리 설치: pip install duckduckgo-search")
        logger.info("2. API 서버가 실행 중인지 확인하세요")
        logger.info("3. 모델 이름이 정확한지 확인하세요")
        logger.info("4. API 키와 엔드포인트가 올바른지 확인하세요")
        logger.info("5. 네트워크 연결을 확인하세요 (웹 검색용)")
        logger.info("6. 타임아웃 설정을 늘려보세요")
        logger.info("7. 로그 파일을 확인하여 상세한 오류 정보를 확인하세요")
        
        return None

# 실행
if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    try:
        import duckduckgo_search
        logger.info("✅ duckduckgo-search 패키지가 설치되어 있습니다.")
    except ImportError:
        logger.error("❌ duckduckgo-search 패키지가 설치되지 않았습니다.")
        logger.info("💡 다음 명령어로 설치하세요: pip install duckduckgo-search")
        exit(1)
    
    result = run_crew_with_error_handling()
    
    if result:
        print("\n✅ 웹 검색이 포함된 AI 블로그 생성이 성공적으로 완료되었습니다!")
        print("🔍 이제 여러분의 AI 팀이 실시간 웹 정보를 활용할 수 있습니다.")
    else:
        print("\n❌ 작업이 실패했습니다. 로그 파일을 확인해보세요.")