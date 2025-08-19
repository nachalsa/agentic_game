import os
import time
import logging
import requests
from datetime import datetime
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import re
import random
from duckduckgo_search import DDGS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'crewai_duckduckgo_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 크루 AI 텔레메트리 비활성화
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# 환경 변수
MODEL_NAME = os.getenv("DEFAULT_LLM", "mistral")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://192.168.100.26:11434")
API_KEY = os.getenv("DEFAULT_API_KEY", "ollama")

# URL 정규화
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# ===== DuckDuckGo 검색 도구 =====
def safe_delay():
    time.sleep(random.uniform(1, 3))

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """
    DuckDuckGo 검색을 수행하고 결과를 반환합니다.
    
    Args:
        query: 검색할 키워드
    
    Returns:
        str: 검색 결과
    """
    try:
        logger.info(f"DuckDuckGo 검색: '{query}'")
        
        # DuckDuckGo 검색 수행
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                region='kr-ko',  # 한국 지역 설정
                safesearch='moderate',
                max_results=8
            ))
        
        if not results:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
        
        # 결과 포맷팅
        formatted_results = f"🔍 '{query}' 검색 결과:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', '제목 없음')
            url = result.get('href', '')
            body = result.get('body', '')
            
            formatted_results += f"{i}. **{title}**\n"
            if body:
                # 본문을 적절한 길이로 자르기
                body_preview = body[:200] + "..." if len(body) > 200 else body
                formatted_results += f"   📄 {body_preview}\n"
            formatted_results += f"   🔗 {url}\n\n"
            
            # 요청 간 딜레이
            if i < len(results):
                safe_delay()
        
        logger.info(f"검색 완료: {len(results)}개 결과")
        return formatted_results
        
    except Exception as e:
        error_msg = f"DuckDuckGo 검색 오류: {str(e)}"
        logger.error(error_msg)
        return error_msg

# ===== 간단한 3단계 에이전트 =====
def create_agents():
    
    # 검색 전략가
    search_strategist = Agent(
        role='검색 전략가',
        goal='주제를 분석하여 효과적인 영어 검색 쿼리 생성',
        backstory='다양한 관점에서 검색 키워드를 생성하는 전문가',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}"
    )

    # 정보 수집가
    information_gatherer = Agent(
        role='정보 수집가',
        goal='DuckDuckGo 검색을 통해 정보를 수집하고 한국어로 정리',
        backstory='웹에서 최신 정보를 수집하여 체계적으로 정리하는 전문가',
        verbose=True,
        allow_delegation=False,
        tools=[duckduckgo_search],
        llm=f"openai/{MODEL_NAME}"
    )

    # 콘텐츠 작성가
    content_creator = Agent(
        role='콘텐츠 작성가',
        goal='수집된 정보로 고품질 한국어 보고서 작성',
        backstory='정보를 바탕으로 흥미로운 보고서를 작성하는 전문가',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}"
    )
    
    return search_strategist, information_gatherer, content_creator

def create_tasks(search_strategist, information_gatherer, content_creator, research_topic):
    
    # 1단계: 검색 전략 수립
    strategy_task = Task(
        description=f'''주제: "{research_topic}"

이 주제에 대한 효과적인 영어 검색 쿼리를 6-8개 생성하세요.

다음 관점들을 포함하세요:
1. 최신 기술/발전사항 (latest, trends, developments)
2. 전문가 분석 (expert analysis, research, study)
3. 실제 사례 (case studies, applications, examples)
4. 미래 전망 (future, outlook, predictions)

출력 형식:
1. {research_topic} latest trends 2024
2. {research_topic} expert analysis
3. {research_topic} case studies
...

모든 쿼리는 영어로 작성하세요.''',
        
        expected_output='6-8개의 영어 검색 쿼리 리스트',
        agent=search_strategist
    )
    
    # 2단계: 정보 수집
    research_task = Task(
        description='''1단계에서 생성된 검색 쿼리들로 DuckDuckGo Search를 수행하세요.

수행 방법:
1. 각 쿼리를 DuckDuckGo Search 도구로 검색
2. 검색 결과에서 핵심 정보 추출
3. 총 6-8회 검색 수행
4. 모든 결과를 한국어로 종합 정리

검색 후 수집된 모든 정보를 체계적으로 정리하세요.''',
        
        expected_output='DuckDuckGo 검색으로 수집된 정보의 종합 보고서 (한국어, 600-800단어)',
        agent=information_gatherer,
        context=[strategy_task]
    )

    # 3단계: 콘텐츠 작성
    content_task = Task(
        description=f'''"{research_topic}: 현실이 된 미래 기술들" 제목의 한국어 보고서를 작성하세요.

요구사항:
- 분량: 900-1200단어
- 언어: 한국어
- 구조: 도입부 → 주요 내용 → 실생활 영향 → 미래 전망 → 결론

포함 요소:
1. 흥미로운 도입부
2. 검색에서 발견된 구체적 정보
3. 실용적 인사이트
4. 미래 전망

검색된 최신 정보를 적극 활용하여 작성하세요.''',
        
        expected_output='900-1200단어의 완성된 한국어 보고서',
        agent=content_creator,
        context=[research_task]
    )
    
    return strategy_task, research_task, content_task

def save_result(result, research_topic):
    if not result:
        return None
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_topic = re.sub(r'[^\w\s-]', '', research_topic.replace(' ', '_'))
    filename = f"duckduckgo_search_report_{safe_topic}_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {research_topic}\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"검색 방식: DuckDuckGo 검색\n")
            f.write(f"AI 모델: {MODEL_NAME}\n\n")
            f.write("---\n\n")
            f.write(str(result))
        
        logger.info(f"결과 저장: {filename}")
        return filename
    except Exception as e:
        logger.error(f"저장 실패: {e}")
        return None

def run_crew(research_topic="2025년 최신 AI 트렌드"):
    try:
        logger.info("=" * 50)
        logger.info("🚀 CrewAI DuckDuckGo 검색 시스템 시작")
        logger.info(f"📋 주제: {research_topic}")
        logger.info("=" * 50)
        
        # LiteLLM 설정
        litellm.api_base = API_BASE_URL
        litellm.api_key = API_KEY
        litellm.drop_params = True
        
        # 에이전트 및 작업 생성
        search_strategist, information_gatherer, content_creator = create_agents()
        strategy_task, research_task, content_task = create_tasks(
            search_strategist, information_gatherer, content_creator, research_topic
        )
        
        # 크루 실행
        crew = Crew(
            agents=[search_strategist, information_gatherer, content_creator],
            tasks=[strategy_task, research_task, content_task],
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("🎯 작업 시작: 검색 전략 → 정보 수집 → 보고서 작성")
        result = crew.kickoff()
        
        # 결과 저장
        saved_file = save_result(result, research_topic)
        
        logger.info("🎉 작업 완료!")
        print(f"\n📄 생성된 보고서:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        if saved_file:
            logger.info(f"📁 파일 저장: {saved_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 실행 오류: {e}", exc_info=True)
        return None

# 실행부
if __name__ == "__main__":
    # 주제 설정 (자유롭게 변경 가능)
    topics = [
        "2025년 최신 AI 트렌드",
        "현대 건축 혁신 기술",
        "지속가능한 에너지 기술",
        "미래 교통 혁신",
        "블록체인 기술 발전"
    ]
    
    # 원하는 주제 선택 (0-4)
    selected_topic = topics[0]
    
    print(f"🎯 연구 주제: {selected_topic}")
    print("🔍 DuckDuckGo 검색으로 최신 정보를 수집합니다...")
    
    # 의존성 확인
    try:
        from duckduckgo_search import DDGS
        print("✅ duckduckgo-search 라이브러리 확인됨")
    except ImportError:
        print("❌ duckduckgo-search 라이브러리가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install duckduckgo-search")
        exit(1)
    
    result = run_crew(selected_topic)
    
    if result:
        print(f"\n✅ '{selected_topic}' 연구 완료!")
        print("📊 DuckDuckGo 검색을 통해 고품질 정보를 수집했습니다.")
    else:
        print("\n❌ 작업 실패. 로그를 확인해보세요.")