import os
import logging
from datetime import datetime
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import argparse
import re
import requests
from urllib.parse import urljoin, urlparse
import time

from ddgs import DDGS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'research_crew_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 설정
load_dotenv()
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# 설정 클래스
class ResearchConfig:
    """리서치 설정을 관리하는 클래스"""
    
    def __init__(self, 
                 topic: str,
                 search_queries_count: int = 5,
                 max_pages_per_query: int = 3,
                 word_count_range: tuple = (700, 900),
                 language: str = "한국어",
                 report_type: str = "블로그"):
        self.topic = topic
        self.search_queries_count = search_queries_count
        self.max_pages_per_query = max_pages_per_query
        self.word_count_range = word_count_range
        self.language = language
        self.report_type = report_type
        
        # 파일명용 안전한 토픽명 생성
        self.safe_topic = re.sub(r'[^\w\s-]', '', topic.replace(' ', '_'))[:50]

# 환경 변수
MODEL_NAME = os.getenv("DEFAULT_LLM", "cpatonn/Devstral-Small-2507-AWQ")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://localhost:54321")
API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "900"))

if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# 통합 웹 검색 도구
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """웹에서 정보를 검색하고 전체 페이지 내용을 추출하는 통합 도구"""
    try:
        logger.info(f"🔍 통합 웹 검색 시작: '{query}'")
        
        # 1단계: 웹 검색
        ddgs = DDGS()
        search_results = ddgs.text(query=query, region='wt-wt', safesearch='moderate', max_results=5)
        
        if not search_results:
            logger.warning(f"⚠️ '{query}' 검색 결과 없음")
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
        
        # 중복 URL 제거
        unique_urls = []
        seen_urls = set()
        
        for result in search_results:
            url = result.get('href', '')
            title = result.get('title', '제목 없음')
            
            if url and url not in seen_urls:
                unique_urls.append({'url': url, 'title': title})
                seen_urls.add(url)
        
        if not unique_urls:
            return f"'{query}'에 대한 유효한 URL을 찾을 수 없습니다."
        
        # 2단계: 페이지 크롤링 및 텍스트 추출
        extracted_contents = []
        max_pages = min(3, len(unique_urls))  # 최대 3개 페이지만 처리
        
        for i, item in enumerate(unique_urls[:max_pages]):
            url = item['url']
            title = item['title']
            
            try:
                logger.info(f"📄 페이지 크롤링 중 ({i+1}/{max_pages}): {url}")
                
                # 페이지 다운로드
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # 3단계: trafilatura로 텍스트 추출
                try:
                    import trafilatura
                    
                    # HTML에서 깨끗한 텍스트 추출
                    extracted_text = trafilatura.extract(
                        response.text,
                        include_comments=False,
                        include_tables=True,
                        include_images=False,
                        output_format='text'
                    )
                    
                    if extracted_text and len(extracted_text.strip()) > 100:
                        # 텍스트 정제
                        clean_text = extracted_text.strip()
                        # 여러 줄바꿈을 2개로 제한
                        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
                        # 너무 긴 텍스트는 첫 2000자만 사용
                        if len(clean_text) > 2000:
                            clean_text = clean_text[:2000] + "..."
                        
                        extracted_contents.append({
                            'title': title,
                            'url': url,
                            'content': clean_text
                        })
                        logger.info(f"✅ 텍스트 추출 성공: {len(clean_text)}자")
                    else:
                        logger.warning(f"⚠️ 텍스트 추출 실패 또는 내용 부족: {url}")
                        
                except ImportError:
                    logger.error("❌ trafilatura 라이브러리가 설치되지 않았습니다")
                    # 기본 HTML 태그 제거 방식으로 폴백
                    clean_text = re.sub(r'<[^>]+>', '', response.text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    if len(clean_text) > 500:
                        extracted_contents.append({
                            'title': title,
                            'url': url,
                            'content': clean_text[:1000] + "..."
                        })
                
                # 요청 간 지연
                time.sleep(1)
                
            except requests.RequestException as e:
                logger.warning(f"⚠️ 페이지 다운로드 실패 {url}: {str(e)}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ 페이지 처리 오류 {url}: {str(e)}")
                continue
        
        # 4단계: 결과 포맷팅
        if not extracted_contents:
            return f"'{query}' 검색 결과에서 텍스트를 추출할 수 없었습니다."
        
        formatted_result = f"🔍 '{query}' 검색 및 텍스트 추출 결과:\n\n"
        
        for i, content in enumerate(extracted_contents, 1):
            formatted_result += f"📄 {i}. {content['title']}\n"
            formatted_result += f"🔗 출처: {content['url']}\n"
            formatted_result += f"📝 내용:\n{content['content']}\n"
            formatted_result += "-" * 80 + "\n\n"
        
        logger.info(f"✅ 통합 검색 완료: {len(extracted_contents)}개 페이지에서 텍스트 추출")
        return formatted_result
        
    except Exception as e:
        error_msg = f"❌ 통합 웹 검색 오류: {str(e)}"
        logger.error(error_msg)
        return error_msg

# 주제별 프리셋 (동일)
RESEARCH_PRESETS = {
    "ai": "2025년 최신 AI 트렌드",
    "blockchain": "2025년 블록체인 기술 발전", 
    "climate": "지속가능한 기후 기술 혁신",
    "health": "디지털 헬스케어 기술 트렌드",
    "fintech": "핀테크 산업 최신 동향",
    "architecture": "현대 건축 기술 혁신",
    "education": "교육 기술 디지털 전환",
    "energy": "재생 에너지 기술 발전",
    "space": "우주 기술 및 탐사 동향",
    "food": "푸드테크 산업 혁신"
}

def get_preset_topic(preset_name: str) -> str:
    """프리셋 주제 반환 (없으면 입력값 그대로 반환)"""
    return RESEARCH_PRESETS.get(preset_name.lower(), preset_name)

# 범용 AI 리서치 크루 클래스 (기존 이름 유지)
class UniversalResearchCrew:
    """모든 주제에 대해 리서치 보고서를 생성하는 AI 크루 시스템"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.setup_environment()
    
    def setup_environment(self):
        """LiteLLM 환경 설정"""
        litellm.api_base = API_BASE_URL
        litellm.api_key = API_KEY
        litellm.drop_params = True
        logger.info("환경 설정 완료")
    
    def create_agents(self):
        """에이전트 생성 (기존 구조 유지)"""
        
        planner = Agent(
            role='연구 계획 전문가',
            goal=f'{self.config.topic}에 대한 효과적인 웹 검색 전략 수립',
            backstory='''다양한 주제를 체계적으로 분석하여 최적의 검색 쿼리를 생성하는 전략가입니다. 
            복잡한 주제를 핵심 질문으로 분해하고, 최신 정보를 얻을 수 있는 검색어를 설계합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=f"openai/{MODEL_NAME}",
            max_tokens=1024,
            temperature=0.6
        )

        researcher = Agent(
            role='전문 리서치 분석가',
            goal=f'{self.config.topic}에 대한 종합적이고 심층적인 정보 수집 및 분석',
            backstory='''통합 웹 검색 도구를 사용하여 실시간 정보를 수집하고, 
            웹페이지 전체 내용을 분석하여 신뢰할 수 있는 인사이트를 도출하는 숙련된 연구 전문가입니다.''',
            verbose=True,
            allow_delegation=False,
            tools=[web_search_tool],
            llm=f"openai/{MODEL_NAME}",
            max_tokens=2000,
            temperature=0.7
        )

        writer = Agent(
            role='전문 콘텐츠 작가',
            goal=f'{self.config.topic}에 대한 매력적이고 유익한 {self.config.report_type} 작성',
            backstory=f'''복잡한 정보를 {self.config.language}로 명확하고 매력적으로 전달하는 전문 작가입니다. 
            다양한 분야의 최신 정보를 독자가 이해하기 쉽고 실용적인 콘텐츠로 변환합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=f"openai/{MODEL_NAME}",
            max_tokens=2000,
            temperature=0.8
        )
        
        return planner, researcher, writer
    
    def create_tasks(self, planner, researcher, writer):
        """태스크 생성 (기존 구조 유지, 내용만 개선)"""
        
        # 1. 검색 계획 수립 (동일)
        planning_task = Task(
            description=f'''"{self.config.topic}"에 대한 포괄적인 연구를 수행해야 합니다.
            
            이 주제를 다음 관점에서 분석하여 {self.config.search_queries_count}개의 구체적이고 효과적인 영어 웹 검색 쿼리를 생성하세요:
            
            1. 최신 동향 및 발전사항 (latest trends, developments, innovations)
            2. 전문가 분석 및 연구 결과 (expert analysis, research, studies)  
            3. 실제 적용 사례 및 케이스 스터디 (case studies, applications, examples)
            4. 미래 전망 및 예측 (future outlook, predictions, forecasts)
            5. 산업별 영향 및 활용 (industry impact, implementation)
            
            **중요한 형식 요구사항:**
            - 각 쿼리는 서로 다른 키워드를 사용해야 합니다
            - 정확히 다음 형식으로 출력하세요:
            SEARCH_QUERY_1: "첫 번째 검색어"
            SEARCH_QUERY_2: "두 번째 검색어"  
            SEARCH_QUERY_3: "세 번째 검색어"
            SEARCH_QUERY_4: "네 번째 검색어"
            SEARCH_QUERY_5: "다섯 번째 검색어"
            
            각 쿼리는 명확하고 독립적으로 검색 가능해야 하며, 2024-2025년 최신 정보를 얻을 수 있도록 구성하세요.''',
            
            expected_output=f'''정확히 다음 형식의 {self.config.search_queries_count}개 영어 검색 쿼리:
            SEARCH_QUERY_1: "query1"
            SEARCH_QUERY_2: "query2"
            SEARCH_QUERY_3: "query3"
            SEARCH_QUERY_4: "query4"  
            SEARCH_QUERY_5: "query5"
            주제: {self.config.topic}에 최적화된 서로 다른 검색어들''',
            agent=planner
        )
        
        # 2. 정보 수집 (통합 도구 사용으로 업데이트)
        research_task = Task(
            description=f'''이전 단계에서 생성된 검색 쿼리 목록을 활용하여 "{self.config.topic}"에 대한 심층 웹 검색 및 텍스트 추출을 수행합니다.

            **필수 수행 절차:**
            1. 이전 Task 결과에서 "SEARCH_QUERY_1:", "SEARCH_QUERY_2:" 등의 형식으로 된 검색 쿼리들을 찾아 추출합니다.
            2. 각 SEARCH_QUERY_X에서 따옴표 안의 검색어만 추출합니다.
            3. 추출된 **모든 검색어를 하나씩 순서대로** 'Web Search Tool'을 사용하여 검색합니다.
            4. 통합 웹 검색 도구가 자동으로 다음을 수행합니다:
               - 웹 검색 실행
               - 상위 3개 페이지 크롤링
               - 전체 HTML에서 trafilatura로 깨끗한 텍스트 추출
               - 정제된 텍스트 결과 반환
            5. 검색할 때마다 "🔍 검색 중: X/{self.config.search_queries_count} - [검색어]" 형태로 진행상황을 알려주세요.
            
            **보고서 작성 요구사항:**
            모든 검색 완료 후, 수집된 정보를 바탕으로 다음을 포함한 종합 보고서를 **반드시 {self.config.language}로** 작성하세요:
            - 주요 트렌드 및 발전사항
            - 핵심 통계 및 데이터  
            - 구체적 사례 및 실무 적용
            - 전문가 의견 및 분석
            - 미래 전망
            
            **절대적으로 중요**: 검색 결과가 영어로 나와도 보고서는 **무조건 {self.config.language}로만** 작성해야 합니다.''',
            
            expected_output=f'''"{self.config.topic}"에 대한 주요 인사이트, 최신 통계 및 실제 예시를 포함하는 
            400-500단어 분량의 상세한 연구 요약 보고서 (**반드시 {self.config.language}로 작성**).
            통합 웹 검색 도구로 추출한 전체 페이지 텍스트를 바탕으로 작성.''',
            
            agent=researcher,
            context=[planning_task]
        )

        # 3. 콘텐츠 작성 (동일)
        write_task = Task(
            description=f'''연구 요약 보고서를 바탕으로 "{self.config.topic}"에 대한 
            **반드시 {self.config.language}로만 작성된** {self.config.report_type}을 작성합니다.
            
            **절대적 언어 요구사항:**
            - **모든 내용은 {self.config.language}로만 작성**해야 합니다
            - 제목, 소제목, 본문, 모든 텍스트가 {self.config.language}여야 합니다
            - 영어 단어나 문장은 절대 사용하지 마세요
            
            **구체적 요구사항:**
            - 분량: {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어
            - 매력적인 {self.config.language} 제목과 부제목 사용
            - 명확하고 이해하기 쉬운 {self.config.language} 표현
            - 연구 보고서의 최신 정보를 반영한 현실적인 내용
            - 독자의 관심을 끄는 구성
            - 실제 사례나 구체적 예시 포함
            
            **구조 ({self.config.language}로 작성):**
            1. 흥미로운 도입부
            2. 주요 내용 (연구 결과 기반)
            3. 실생활/산업 적용 사례
            4. 미래 전망
            5. 결론 및 요약
            
            **다시 한번 강조**: 단 한 단어도 영어로 작성하지 말고, 모든 내용을 {self.config.language}로만 작성하세요.
            
            대상 독자: 해당 분야에 관심있는 일반인 및 전문가''',
            
            expected_output=f'''독자 친화적이고 정보가 풍부한 {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어 분량의 
            {self.config.report_type}. **완전히 {self.config.language}로만 작성**되었으며, 
            통합 웹 검색 도구로 추출한 실제 웹페이지 내용을 반영한 현실적이고 유용한 내용 포함.''',
            
            agent=writer,
            context=[research_task]
        )
        
        return planning_task, research_task, write_task
    
    def save_result(self, result):
        """결과를 파일로 저장 (동일)"""
        if not result:
            logger.warning("저장할 결과가 없습니다.")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_report_{self.config.safe_topic}_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {self.config.topic} - 연구 보고서\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"보고서 유형: {self.config.report_type}\n")
                f.write(f"언어: {self.config.language}\n")
                f.write(f"검색 쿼리 수: {self.config.search_queries_count}개\n")
                f.write(f"페이지당 최대 크롤링: {self.config.max_pages_per_query}개\n\n")
                f.write("---\n\n")
                f.write(str(result))
            
            logger.info(f"결과가 {filename}에 저장되었습니다.")
            return filename
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return None
    
    def research(self):
        """메인 리서치 실행 메서드 (동일)"""
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 범용 AI 리서치 크루 시작")
            logger.info(f"📋 주제: {self.config.topic}")
            logger.info(f"📊 보고서 유형: {self.config.report_type}")
            logger.info(f"🔍 통합 웹 검색 도구 사용 (검색+크롤링+텍스트추출)")
            logger.info("=" * 60)
            
            # 에이전트 및 작업 생성
            planner, researcher, writer = self.create_agents()
            planning_task, research_task, write_task = self.create_tasks(planner, researcher, writer)
            
            # 크루 실행
            crew = Crew(
                agents=[planner, researcher, writer],
                tasks=[planning_task, research_task, write_task],
                process=Process.sequential,
                verbose=True,
                max_execution_time=MAX_EXECUTION_TIME
            )
            
            logger.info(f"\n🎯 AI 크루 작업 시작: {self.config.topic}")
            logger.info("예상 소요 시간: 5-8분 (페이지 크롤링 포함)")
            
            result = crew.kickoff()
            
            # 결과 저장
            saved_file = self.save_result(result)
            
            logger.info("\n🎉 크루 작업 완료!")
            print(f"\n📄 생성된 {self.config.report_type}:")
            print("=" * 80)
            print(result)
            print("=" * 80)
            
            if saved_file:
                logger.info(f"\n📁 결과가 '{saved_file}' 파일에 저장되었습니다.")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 리서치 실행 오류: {e}", exc_info=True)
            return None

def main():
    """메인 실행 함수 (기존 동일)"""
    parser = argparse.ArgumentParser(description='범용 AI 리서치 크루 - 통합 웹 검색으로 깊이 있는 보고서 생성')
    parser.add_argument('--topic', '-t', 
                        default='2025년 최신 AI 트렌드', 
                        help='연구 주제 (또는 프리셋: ai, blockchain, health, etc.)')
    parser.add_argument('--queries', '-q', 
                        type=int, default=5, 
                        help='검색 쿼리 개수 (기본값: 5)')
    parser.add_argument('--pages', '-p', 
                        type=int, default=3, 
                        help='쿼리당 크롤링할 최대 페이지 수 (기본값: 3)')
    parser.add_argument('--words', '-w', 
                        default='700,900', 
                        help='단어 수 범위 (예: 700,900)')
    parser.add_argument('--type', '-r', 
                        default='블로그', 
                        help='보고서 유형 (블로그, 보고서, 분석서 등)')
    parser.add_argument('--language', '-l', 
                        default='한국어', 
                        help='출력 언어')
    parser.add_argument('--list-presets', 
                        action='store_true', 
                        help='사용 가능한 프리셋 주제 목록 출력')
    
    args = parser.parse_args()
    
    # 프리셋 목록 출력
    if args.list_presets:
        print("📋 사용 가능한 프리셋 주제:")
        for key, value in RESEARCH_PRESETS.items():
            print(f"  {key}: {value}")
        return
    
    # 설정 생성
    try:
        word_range = tuple(map(int, args.words.split(',')))
        if len(word_range) != 2:
            raise ValueError("단어 수는 '최소,최대' 형식으로 입력하세요")
    except:
        word_range = (700, 900)
        print("⚠️ 단어 수 형식 오류. 기본값 (700,900) 사용")
    
    # 프리셋 확인 및 주제 설정
    topic = get_preset_topic(args.topic)
    
    config = ResearchConfig(
        topic=topic,
        search_queries_count=args.queries,
        max_pages_per_query=args.pages,
        word_count_range=word_range,
        language=args.language,
        report_type=args.type
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
    print(f"📄 쿼리당 크롤링: 최대 {config.max_pages_per_query}개 페이지")
    print(f"📝 목표 단어 수: {config.word_count_range[0]}-{config.word_count_range[1]}단어")
    
    # 리서치 실행
    crew = UniversalResearchCrew(config)
    result = crew.research()
    
    if result:
        print(f"\n✅ '{config.topic}' 연구 보고서가 성공적으로 생성되었습니다!")
    else:
        print(f"\n❌ 작업 실패. 로그를 확인해보세요.")

def run_default():
    """기본 실행 함수 (동일)"""
    print("🚀 범용 AI 리서치 크루 시작!")
    print("📋 기본 주제로 보고서를 생성합니다...")
    
    # 기본 설정으로 연구 실행
    config = ResearchConfig(
        topic="2025년 최신 AI 트렌드",
        search_queries_count=5,
        max_pages_per_query=3,
        word_count_range=(700, 900),
        language="한국어",
        report_type="블로그"
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
    print(f"📄 쿼리당 크롤링: 최대 {config.max_pages_per_query}개 페이지")
    print(f"📝 목표 단어 수: {config.word_count_range[0]}-{config.word_count_range[1]}단어")
    print("\n" + "="*60)
    
    # 리서치 실행
    crew = UniversalResearchCrew(config)
    result = crew.research()
    
    if result:
        print(f"\n✅ '{config.topic}' 연구 보고서가 성공적으로 생성되었습니다!")
        print("💡 다른 주제로 연구하려면: python script.py --topic '원하는 주제'")
    else:
        print(f"\n❌ 작업 실패. 로그를 확인해보세요.")
    
    return result

if __name__ == "__main__":
    # 명령행 인자가 있으면 CLI 모드, 없으면 기본 실행
    import sys
    if len(sys.argv) > 1:
        main()  # CLI 모드
    else:
        run_default()  # 기본 실행 모드