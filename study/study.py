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
import json
import re

# Web scraping libraries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'crewai_simple_{datetime.now().strftime("%Y%m%d")}.log'),
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
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "600"))

# URL 정규화
if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# ===== 구글 직접 검색 도구 =====

# User-Agent 리스트 (차단 방지)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def safe_delay():
    """차단 방지를 위한 랜덤 딜레이"""
    delay = random.uniform(2, 5)
    time.sleep(delay)
    logger.info(f"대기 중... ({delay:.1f}초)")

def get_random_headers():
    """랜덤 헤더 생성"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def extract_text_from_url(url, max_chars=1500):
    """URL에서 본문 텍스트 추출"""
    try:
        logger.info(f"본문 추출 중: {url}")
        headers = get_random_headers()
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 불필요한 태그들 제거
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # 본문 텍스트 추출
        text = soup.get_text(separator=' ', strip=True)
        
        # 텍스트 정리
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = ' '.join(lines)
        
        # 길이 제한
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "..."
        
        return clean_text
        
    except Exception as e:
        logger.error(f"본문 추출 실패 {url}: {e}")
        return ""

def parse_google_results(html_content):
    """구글 검색 결과에서 링크들 추출"""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    
    # 구글 검색 결과 링크 찾기
    for g in soup.find_all('div', class_='g'):
        try:
            # 제목
            title_elem = g.find('h3')
            title = title_elem.get_text() if title_elem else "제목 없음"
            
            # URL
            link_elem = g.find('a')
            if link_elem and 'href' in link_elem.attrs:
                url = link_elem['href']
                
                # 구글 내부 링크 제외
                if url.startswith('http') and 'google.com' not in url:
                    results.append({
                        'title': title,
                        'url': url
                    })
                    
        except Exception as e:
            logger.error(f"결과 파싱 오류: {e}")
            continue
    
    return results[:5]  # 상위 5개만

@tool("Google Search")
def google_search(query: str) -> str:
    """
    구글 직접 검색을 수행합니다.
    
    Args:
        query: 검색할 키워드 (영어 권장)
    
    Returns:
        str: 검색 결과 및 본문 내용
    """
    try:
        logger.info(f"구글 검색 시작: '{query}'")
        
        # 구글 검색 URL
        search_url = f"https://www.google.com/search?q={query}&num=10"
        
        # 검색 요청
        headers = get_random_headers()
        response = requests.get(search_url, headers=headers, timeout=15)
        
        if response.status_code == 429:
            logger.warning("요청 제한 감지, 긴 대기 후 재시도")
            time.sleep(30)
            response = requests.get(search_url, headers=headers, timeout=15)
        
        response.raise_for_status()
        
        # 검색 결과 파싱
        results = parse_google_results(response.text)
        
        if not results:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
        
        # 결과 포맷팅 시작
        formatted_results = f"🔍 '{query}' 구글 검색 결과:\n\n"
        
        # 각 결과에서 본문 추출
        for i, result in enumerate(results, 1):
            title = result['title']
            url = result['url']
            
            # 딜레이 (첫 번째 제외)
            if i > 1:
                safe_delay()
            
            # 본문 내용 추출
            content = extract_text_from_url(url)
            
            if content:
                formatted_results += f"{i}. **{title}**\n"
                formatted_results += f"   📄 {content[:300]}{'...' if len(content) > 300 else ''}\n"
                formatted_results += f"   🔗 {url}\n\n"
            else:
                # 본문 추출 실패 시에도 기본 정보는 포함
                formatted_results += f"{i}. **{title}**\n"
                formatted_results += f"   🔗 {url}\n\n"
        
        logger.info(f"구글 검색 완료: {len(results)}개 결과 처리")
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        error_msg = f"구글 검색 연결 오류: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"구글 검색 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return error_msg

# ===== 유틸리티 함수들 =====
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

def test_litellm_connection():
    def _test():
        logger.info("LiteLLM 연결 테스트 중...")
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": "user", "content": "안녕하세요"}],
            api_base=API_BASE_URL,
            api_key=API_KEY,
            temperature=0.7,
            max_tokens=100,
            timeout=TIMEOUT,
            drop_params=True
        )
        logger.info("LiteLLM 테스트 성공")
        return True
    return retry_with_backoff(_test)

def test_google_search():
    try:
        logger.info("구글 검색 기능 테스트 중...")
        test_result = google_search.func("AI trends 2025")
        if "구글 검색 결과" in test_result or "오류" in test_result:
            logger.info("✅ 구글 검색 테스트 성공")
            return True
        else:
            logger.warning("⚠️ 구글 검색 결과가 예상과 다릅니다")
            return False
    except Exception as e:
        logger.error(f"❌ 구글 검색 테스트 실패: {e}")
        return False

def setup_litellm_global_config():
    litellm.api_base = API_BASE_URL
    litellm.api_key = API_KEY
    litellm.drop_params = True
    logger.info("LiteLLM 전역 설정 완료")

# ===== 3단계 에이전트 생성 =====
def create_simple_agents():
    
    # 1단계: 검색 전략가
    search_strategist = Agent(
        role='검색 전략가',
        goal='주제를 분석하여 효과적인 영어 검색 쿼리들을 생성',
        backstory='''당신은 주제를 분석하여 포괄적인 정보 수집을 위한 
        검색 전략을 수립하는 전문가입니다. 기술, 비즈니스, 규제, 미래전망 등 
        다양한 관점에서 검색 키워드를 생성합니다.''',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}",
        max_tokens=800,
        temperature=0.6
    )

    # 2단계: 정보 수집가  
    information_gatherer = Agent(
        role='정보 수집가',
        goal='웹 검색을 통해 최신 정보를 수집하고 한국어로 정리',
        backstory='''당신은 주어진 검색 쿼리를 사용하여 웹에서 최신 정보를 
        수집하는 전문가입니다. 검색 결과를 분석하여 핵심 정보를 추출하고 
        한국어로 체계적으로 정리합니다.''',
        verbose=True,
        allow_delegation=False,
        tools=[google_search],
        llm=f"openai/{MODEL_NAME}",
        max_tokens=2000,
        temperature=0.5
    )

    # 3단계: 콘텐츠 작성가
    content_creator = Agent(
        role='콘텐츠 작성가',
        goal='수집된 정보를 바탕으로 고품질 한국어 블로그 포스트 작성',
        backstory='''당신은 수집된 정보를 바탕으로 독자가 흥미롭게 읽을 수 있는 
        고품질 블로그 포스트를 작성하는 전문가입니다. 최신 정보와 실용적인 
        인사이트를 제공하여 독자에게 가치를 전달합니다.''',
        verbose=True,
        allow_delegation=False,
        llm=f"openai/{MODEL_NAME}",
        max_tokens=3000,
        temperature=0.7
    )
    
    return search_strategist, information_gatherer, content_creator

# ===== 3단계 작업 생성 =====
def create_simple_tasks(search_strategist, information_gatherer, content_creator, research_topic):
    
    # 1단계: 검색 전략 수립
    strategy_task = Task(
        description=f'''주제: "{research_topic}"

이 주제에 대한 포괄적인 연구를 위해 효과적인 영어 검색 쿼리를 생성하세요.

다음 관점들을 고려하여 검색 쿼리를 만드세요:
1. 최신 기술 발전사항 (latest technology, breakthrough, innovation)
2. 비즈니스 동향 (business trends, market adoption, industry impact)  
3. 규제 및 정책 (regulation, policy, governance, ethics)
4. 미래 전망 (future outlook, predictions, forecasts)
5. 실제 사례 (case studies, real-world applications)

각 관점별로 1-2개씩, 총 6-8개의 구체적인 영어 검색 쿼리를 생성하세요.
모든 쿼리는 반드시 영어로 작성되어야 합니다.

출력 형식:
1. AI breakthrough technology 2025
2. generative AI business adoption trends
3. AI regulation policy developments 2025
...''',
        
        expected_output='''6-8개의 구체적인 영어 검색 쿼리 리스트.
각 쿼리는 번호와 함께 명확하게 나열되어야 함.
모든 쿼리는 영어로 작성되어야 함.''',
        
        agent=search_strategist
    )
    
    # 2단계: 정보 수집
    research_task = Task(
        description='''1단계에서 생성된 검색 쿼리들을 사용하여 웹 검색을 수행하세요.

수행 방법:
1. 제공된 검색 쿼리를 하나씩 Web Search 도구로 검색
2. 각 검색 결과에서 핵심 정보 추출
3. 관련성 높은 추가 쿼리가 있다면 추가 검색 수행
4. 총 6-8회의 검색을 통해 다양한 정보 수집

주의사항:
        - 반드시 Google Search 도구를 사용하여 실제 검색 수행
- 각 검색 후 결과를 간단히 요약
- 최신 정보와 구체적인 데이터에 집중
- 모든 정리는 한국어로 작성

검색이 완료되면 수집된 모든 정보를 종합하여 체계적으로 정리하세요.''',
        
        expected_output='''웹 검색을 통해 수집된 최신 정보의 종합 보고서 (한국어).
각 검색 결과의 핵심 내용과 출처를 포함.
600-800단어 분량의 체계적인 정보 정리.''',
        
        agent=information_gatherer,
        context=[strategy_task]
    )

    # 3단계: 콘텐츠 작성
    content_task = Task(
        description=f'''수집된 정보를 바탕으로 "{research_topic}: 현실이 된 미래 기술들" 제목의 
고품질 한국어 블로그 포스트를 작성하세요.

요구사항:
- 분량: 900-1200단어
- 언어: 한국어
- 어조: 전문적이면서도 이해하기 쉽게
- 구조: 도입부 → 주요 트렌드 분석 → 실생활 영향 → 미래 전망 → 결론

포함 요소:
1. 흥미로운 도입부 (최신 사례나 놀라운 발전사항)
2. 검색에서 발견된 구체적인 데이터와 사례
3. 독자에게 실용적인 인사이트 제공
4. 미래에 대한 전문가적 전망
5. 독자가 준비할 수 있는 조언

반드시 한국어로 작성하고, 연구에서 수집된 최신 정보를 적극 활용하세요.''',
        
        expected_output='''900-1200단어 분량의 완성된 한국어 블로그 포스트.
수집된 최신 정보를 바탕으로 한 실용적이고 통찰력 있는 내용.
독자가 끝까지 읽고 싶어하는 흥미로운 구성.''',
        
        agent=content_creator,
        context=[research_task]
    )
    
    return strategy_task, research_task, content_task

def save_result(result, research_topic):
    """결과 저장"""
    if not result:
        logger.warning("저장할 결과가 없습니다.")
        return None
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_topic = re.sub(r'[^\w\s-]', '', research_topic.replace(' ', '_'))
    filename = f"simple_ai_blog_{safe_topic}_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {research_topic}\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"AI 모델: {MODEL_NAME}\n\n")
            f.write("---\n\n")
            f.write(str(result))
        
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        return filename
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return None

def run_simple_crew(research_topic="2025년 최신 AI 트렌드"):
    """간단한 3단계 크루 실행"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 간단한 CrewAI 시스템 시작")
        logger.info(f"📋 연구 주제: {research_topic}")
        logger.info("=" * 60)
        
        # 설정 및 연결 테스트
        logger.info(f"🔧 설정: {MODEL_NAME} @ {API_BASE_URL}")
        test_litellm_connection()
        test_google_search()
        setup_litellm_global_config()
        
        # 에이전트 및 작업 생성
        logger.info("\n🤖 3단계 AI 팀 구성 중...")
        search_strategist, information_gatherer, content_creator = create_simple_agents()
        strategy_task, research_task, content_task = create_simple_tasks(
            search_strategist, information_gatherer, content_creator, research_topic
        )
        
        # 크루 생성 및 실행
        simple_crew = Crew(
            agents=[search_strategist, information_gatherer, content_creator],
            tasks=[strategy_task, research_task, content_task],
            process=Process.sequential,
            verbose=True,
            max_execution_time=MAX_EXECUTION_TIME,
            memory=False,
            max_rpm=30  # RPM 증가
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 3단계 AI 팀 작업 시작")
        logger.info("1️⃣ 검색 전략 수립 → 2️⃣ 정보 수집 → 3️⃣ 콘텐츠 작성")
        logger.info("=" * 60)
        
        result = simple_crew.kickoff()
        
        # 결과 저장 및 출력
        saved_file = save_result(result, research_topic)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 작업 완료!")
        logger.info("=" * 60)
        print(f"\n📄 생성된 콘텐츠:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        if saved_file:
            logger.info(f"\n📁 결과가 '{saved_file}' 파일에 저장되었습니다.")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 크루 실행 중 오류 발생: {e}", exc_info=True)
        return None

# 실행부
if __name__ == "__main__":
    # 사용자 정의 주제
    custom_topic = "2025년 최신 AI 트렌드"
    
    print(f"🎯 연구 주제: {custom_topic}")
    print("🤖 간단한 3단계 AI 팀이 작업을 시작합니다...")
    
    result = run_simple_crew(custom_topic)
    
    if result:
        print(f"\n✅ '{custom_topic}' 연구가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업 실패. 로그를 확인해보세요.")