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
        logging.FileHandler(f'crewai_dynamic_{datetime.now().strftime("%Y%m%d")}.log'),
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
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "600")) # 실행 시간 증가

# URL 정규화
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# ===== 웹 검색 툴 (변경 없음) =====
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """
    웹에서 정보를 검색하는 도구입니다. 최신 정보나 트렌드를 찾을 때 사용하세요.
    
    Args:
        query: 검색할 키워드나 질문 (예: "2025년 AI 트렌드", "최신 생성형 AI 발전사항")
    
    Returns:
        str: 검색 결과 목록 (제목, 설명, 링크 포함)
    """
    try:
        logger.info(f"웹 검색 시작: '{query}'")
        ddgs = DDGS()
        results = ddgs.text(
            query=query, 
            region='wt-wt',
            safesearch='moderate', 
            max_results=5
        )
        if not results:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
        
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

# ===== 기존 함수들 (변경 없음) =====
def retry_with_backoff(func, max_retries=3, base_delay=1):
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
    session = requests.Session()
    try:
        yield session
    finally:
        session.close()

def test_litellm_connection():
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
    try:
        logger.info("웹 검색 기능 테스트 중...")
        test_result = web_search_tool.func("AI 트렌드 2025")
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
    litellm.api_base = API_BASE_URL
    litellm.api_key = API_KEY
    litellm.drop_params = True
    os.environ['LITELLM_LOG'] = 'INFO'
    logger.info("LiteLLM 전역 설정 완료")

# ===== [수정 없음] 에이전트 생성 함수 =====
def create_agents():
    """
    동적 검색을 위한 에이전트들을 생성합니다.
    - `planner`: 연구 주제를 분석하여 검색 계획(쿼리 목록)을 수립합니다.
    - `researcher`: 계획에 따라 웹 검색을 수행하고 정보를 종합합니다.
    - `writer`: 종합된 정보를 바탕으로 블로그 글을 작성합니다.
    """
    
    # 새로운 에이전트: 연구 계획자
    planner = Agent(
        role='연구 계획 전문가',
        goal='주어진 연구 주제를 분석하여 효과적인 웹 검색 쿼리 목록을 생성',
        backstory='''당신은 복잡한 주제를 핵심적인 질문으로 분해하는 데 능숙한 전략가입니다. 
        당신의 목표는 연구 분석가가 최상의 결과를 얻을 수 있도록 명확하고 간결한 검색 계획을 제공하는 것입니다.''',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}",
        max_tokens=1024,
        temperature=0.6
    )

    # 기존 연구 분석가 (역할 재정의)
    researcher = Agent(
        role='고급 연구 분석가',
        goal='제공된 검색 계획에 따라 최신 AI 트렌드 정보를 수집하고 종합적인 보고서 작성',
        backstory='''당신은 웹 검색을 통해 실시간 정보를 수집하고, 여러 출처의 정보를 비판적으로 분석하여 
        핵심 인사이트를 도출하는 데 특화된 숙련된 연구원입니다.''',
        verbose=True,
        allow_delegation=False,
        tools=[web_search_tool],
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.7
    )

    # 기존 전문 작가 (변경 없음)
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
    
    return planner, researcher, writer

# ===== [수정됨] 작업 생성 함수 =====
def create_tasks(planner, researcher, writer):
    """
    동적 검색을 위한 작업들을 생성합니다.
    """
    
    # 1. 연구 주제를 분석하여 검색 계획을 수립하는 작업
    planning_task = Task(
        description='''2025년 최신 AI 트렌드에 대한 포괄적인 연구를 수행해야 합니다.
        이 목표를 달성하기 위해, 가장 중요하고 관련성 높은 하위 주제들을 식별하세요.

        다음 영역을 고려하여 4~5개의 구체적이고 효과적인 영어 웹 검색 쿼리를 생성하세요:
        - 생성형 AI (Generative AI)의 최신 발전
        - 대규모 언어 모델 (LLM)의 새로운 기능 또는 아키텍처
        - AI 윤리 및 규제 동향
        - 다양한 산업 분야에서의 AI 적용 사례
        
        각 쿼리는 명확하고 독립적으로 검색 가능해야 합니다.''',
        
        expected_output='''연구 목표를 달성하기 위한 4~5개의 영어 검색 쿼리 목록.
        각 쿼리는 한 줄로 명확하게 구분되어야 합니다. (예: `- "query"` 형식)
        예시:
        - "latest breakthroughs in multimodal generative AI 2025"
        - "new architectures for large language models 2025"
        - "global AI ethics and regulation policies 2025"
        - "AI applications in healthcare industry 2025 case studies"
        ''',
        agent=planner
    )
    
    # 2. 생성된 계획에 따라 웹 검색을 수행하고 요약하는 작업
    research_task = Task(
        description='''[이전 단계에서 생성된 검색 쿼리 목록]을 활용하여 2025년 최신 AI 트렌드에 대한 심층 웹 검색을 수행하고, 
        주요 발견 사항을 종합적으로 요약합니다.

        **수행 절차:**
        1. 이전 Task의 결과물에서 제공된 **각각의 검색 쿼리(예: `- "query"` 형식)**를 정확히 추출합니다.
        2. 추출된 **모든 쿼리에 대해** 'Web Search Tool'을 **순서대로 개별적으로 사용**합니다.
           각 검색 후에는 결과를 분석하고, 다음 쿼리로 진행합니다.
           예시:
           Thought: 첫 번째 쿼리 "latest breakthroughs in multimodal generative AI 2025"로 검색해야겠다.
           Action: Web Search Tool
           Action Input: {"query": "latest breakthroughs in multimodal generative AI 2025"}
           Observation: (검색 결과)
           Thought: 이제 두 번째 쿼리 "new architectures for large language models 2025"로 검색해야겠다.
           Action: Web Search Tool
           Action Input: {"query": "new architectures for large language models 2025"}
           Observation: (검색 결과)
           ... 이런 방식으로 **모든 쿼리를 소진**할 때까지 반복합니다.
        3. 모든 검색이 완료되면, 수집된 **모든 정보**를 비판적으로 분석하고, 가장 신뢰할 수 있고 최신 정보를 종합하여
           핵심적인 인사이트, 최신 통계, 그리고 구체적인 예시를 포함한 보고서를 작성합니다.
           (주의: 검색 결과는 영어일 수 있으나, 보고서는 한국어로 핵심 내용을 요약하여 작성되어야 합니다.)
        4. 보고서는 주요 트렌드별로 구조화하여 정리해야 합니다.

        모든 쿼리를 사용하여 충분한 정보를 수집했는지 다시 한번 확인하세요.
        ''',
        
        expected_output='''2025년 AI 트렌드에 대한 주요 통찰력, 최신 통계 및 실제 예시를 포함하는 
        400-500단어 분량의 상세한 연구 요약 보고서 (한국어).
        **동적으로 생성된 모든 쿼리**를 통해 얻은 최신 정보를 바탕으로 작성되어야 합니다.''',
        
        agent=researcher,
        context=[planning_task]  # planning_task의 결과(검색어 목록)를 이 task의 context로 사용
    )

    # 3. 연구 요약을 바탕으로 블로그 글을 작성하는 작업
    write_task = Task(
        description='''연구 요약 보고서를 바탕으로 "2025년 AI 혁명: 현실이 된 미래 기술들"이라는 
        제목의 한국어 블로그 게시물을 작성합니다.
        
        **매우 중요:** 모든 내용은 **한국어**로 작성되어야 합니다. 연구 보고서 내용 중 영어 표현이 있다면, 
        이를 자연스럽고 명확한 한국어로 번역하여 본문에 포함시키세요.
        
        요구사항:
        - 700-900단어 분량
        - 매력적인 도입부와 결론
        - 연구 보고서의 최신 정보를 반영한 현실적인 내용
        - 명확하고 이해하기 쉬운 언어 사용
        - 독자의 관심을 끄는 부제목 활용
        - 2025년 현재 상황을 반영한 실제 사례나 예시 포함
        
        대상 독자: AI에 관심있는 일반 대중 및 비즈니스 전문가''',
        
        expected_output='''독자의 참여를 유도하고 정보가 풍부하며 잘 구성된 700-900단어 분량의 
        블로그 게시물. 모든 내용은 한국어로 작성되었으며, 동적 웹 검색 결과를 반영한 
        현실적이고 유용한 내용 포함.''',
        
        agent=writer,
        context=[research_task]
    )
    
    return planning_task, research_task, write_task

def save_result(result):
    """결과를 파일로 저장"""
    if not result:
        logger.warning("저장할 결과가 없습니다.")
        return None
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ai_trends_blog_dynamic_search_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# AI 트렌드 블로그 포스트 (동적 웹 검색 활용)\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"검색 기능: DuckDuckGo (동적 쿼리 생성)\n\n")
            f.write("---\n\n")
            f.write(str(result))
        
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        return filename
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return None

# ===== [verbose=True로 수정됨] Crew 실행 함수 =====
def run_crew_with_error_handling():
    """에러 처리가 포함된 크루 실행 함수 (동적 웹 검색 포함)"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 CrewAI with Dynamic Web Search 시작")
        logger.info("=" * 60)
        
        # 설정 검증 및 연결 테스트
        logger.info(f"모델: {MODEL_NAME}, API Base: {API_BASE_URL}")
        test_litellm_connection()
        test_web_search()
        setup_litellm_global_config()
        
        # 에이전트 및 작업 생성
        logger.info("\n📝 에이전트 및 작업 설정 중...")
        planner, researcher, writer = create_agents()
        planning_task, research_task, write_task = create_tasks(planner, researcher, writer)
        
        # 크루 생성 및 실행
        crew = Crew(
            agents=[planner, researcher, writer],
            tasks=[planning_task, research_task, write_task],
            process=Process.sequential,
            verbose=True, # ✅ 이 부분을 True로 수정했습니다.
            max_execution_time=MAX_EXECUTION_TIME
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🚀 AI 크루 작업 시작 (동적 웹 검색)")
        logger.info("예상 소요 시간: 3-5분 (계획 수립 및 다중 검색 포함)")
        logger.info("=" * 60)
        
        result = crew.kickoff()
        
        # 결과 저장 및 출력
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
            logger.info("💡 이 파일에는 에이전트가 동적으로 생성한 검색어를 통해 수집된 최신 정보가 포함되어 있습니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 크루 실행 중 오류 발생: {e}", exc_info=True)
        logger.info("\n🔧 문제 해결 방법 제안:")
        logger.info("1. `pip install --upgrade crewai crewai-tools duckduckgo-search` 로 최신 라이브러리를 설치하세요.")
        logger.info("2. 로컬 API 서버가 정상적으로 실행 중인지 확인하세요.")
        logger.info("3. .env 파일의 모델 이름, API 주소, API 키가 올바른지 확인하세요.")
        logger.info("4. 네트워크 연결 상태를 확인하세요.")
        logger.info("5. 로그 파일을 통해 상세한 오류 원인을 파악하세요.")
        return None

# 실행
if __name__ == "__main__":
    try:
        import duckduckgo_search
        logger.info("✅ duckduckgo-search 패키지가 설치되어 있습니다.")
    except ImportError:
        logger.error("❌ duckduckgo-search 패키지가 설치되지 않았습니다.")
        logger.info("💡 다음 명령어로 설치하세요: pip install duckduckgo-search")
        exit(1)
    
    result = run_crew_with_error_handling()
    
    if result:
        print("\n✅ 동적 웹 검색을 포함한 AI 블로그 생성이 성공적으로 완료되었습니다!")
        print("🔍 이제 여러분의 AI 팀이 스스로 계획을 세우고 실시간 정보를 활용할 수 있습니다.")
    else:
        print("\n❌ 작업이 실패했습니다. 로그 파일을 확인해보세요.")