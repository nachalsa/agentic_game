"""
통합된 AI 리서치 크루 - study.py 기반으로 개선된 버전
- improved_research_crew.py
- korean_optimized_crew.py  
- fixed_search_tool.py
의 기능을 통합하여 재사용 가능한 형태로 구성
"""
import os
import sys
import logging
import argparse
import re
import hashlib
from datetime import datetime
from typing import Set, List, Dict, Any
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

# DuckDuckGo Search
from ddgs import DDGS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'unified_research_crew_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 설정
load_dotenv()
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# 환경 변수
MODEL_NAME = os.getenv("DEFAULT_LLM", "mistral-small3.2")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://192.168.100.26:11434")
API_KEY = os.getenv("DEFAULT_API_KEY", "ollama")
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "600"))

if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# 설정 클래스
class ResearchConfig:
    """리서치 설정을 관리하는 클래스"""
    
    def __init__(self, 
                 topic: str,
                 search_queries_count: int = 5,
                 word_count_range: tuple = (700, 900),
                 language: str = "한국어",
                 report_type: str = "블로그",
                 quality_mode: str = "standard"):  # standard, korean_optimized
        self.topic = topic
        self.search_queries_count = search_queries_count
        self.word_count_range = word_count_range
        self.language = language
        self.report_type = report_type
        self.quality_mode = quality_mode
        
        # 파일명용 안전한 토픽명 생성
        self.safe_topic = re.sub(r'[^\w\s-]', '', topic.replace(' ', '_'))[:50]

# 개선된 웹 검색 도구 (fixed_search_tool.py 기반)
_search_history: Set[str] = set()
_search_results_cache: Dict[str, str] = {}

def clear_search_history():
    """검색 히스토리 초기화"""
    global _search_history, _search_results_cache
    _search_history.clear()
    _search_results_cache.clear()

def get_query_hash(query: str) -> str:
    """쿼리의 해시값 생성 (유사한 쿼리 감지용)"""
    normalized_query = query.lower().strip()
    return hashlib.md5(normalized_query.encode()).hexdigest()

@tool("Web Search Tool")
def improved_web_search_tool(query: str) -> str:
    """개선된 웹 검색 도구 - 중복 방지 및 에러 처리 강화"""
    
    # 입력 검증
    if not query or not isinstance(query, str):
        return "❌ 유효하지 않은 검색 쿼리입니다."
    
    query = query.strip()
    if len(query) < 3:
        return "❌ 검색 쿼리가 너무 짧습니다. 최소 3글자 이상 입력해주세요."
    
    # 쿼리 해시 생성
    query_hash = get_query_hash(query)
    
    # 중복 검색 방지
    if query_hash in _search_history:
        if query_hash in _search_results_cache:
            return f"🔄 (캐시됨) {_search_results_cache[query_hash]}"
        else:
            return f"⚠️ 이미 검색한 쿼리입니다: '{query}'. 다른 검색어를 시도해보세요."
    
    # 검색 히스토리에 추가
    _search_history.add(query_hash)
    
    try:
        logger.info(f"🔍 웹 검색 시작: '{query}'")
        
        # DuckDuckGo 검색 실행
        ddgs = DDGS()
        results = ddgs.text(
            query=query, 
            region='wt-wt', 
            safesearch='moderate', 
            max_results=5
        )
        
        if not results:
            error_msg = f"⚠️ '{query}'에 대한 검색 결과를 찾을 수 없습니다."
            logger.warning(error_msg)
            return error_msg
        
        # 결과 포매팅
        formatted_results = format_search_results(query, results)
        
        # 캐시에 저장
        _search_results_cache[query_hash] = formatted_results
        
        logger.info(f"✅ 검색 완료: {len(results)}개 결과")
        return formatted_results
        
    except Exception as e:
        error_msg = f"❌ 검색 오류 발생: {str(e)}"
        logger.error(f"웹 검색 오류 - Query: '{query}', Error: {e}")
        
        # DNS 오류 등 네트워크 문제인 경우 대체 메시지
        if "dns error" in str(e).lower() or "name or service not known" in str(e).lower():
            error_msg += "\n🌐 인터넷 연결을 확인하고 다시 시도해주세요."
        
        return error_msg

def format_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """검색 결과를 포매팅"""
    formatted = f"🔍 '{query}' 검색 결과:\n\n"
    
    # 중복 URL 제거
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get('href', '')
        if url and url not in seen_urls:
            unique_results.append(result)
            seen_urls.add(url)
    
    if not unique_results:
        return f"⚠️ '{query}'에 대한 유효한 검색 결과가 없습니다."
    
    for i, result in enumerate(unique_results, 1):
        title = result.get('title', '제목 없음')
        body = result.get('body', '설명 없음')
        href = result.get('href', '#')
        
        # 본문이 너무 길면 자르기
        if len(body) > 150:
            body = body[:150] + '...'
            
        formatted += f"{i}. **{title}**\n"
        formatted += f"   📄 {body}\n"
        formatted += f"   🔗 {href}\n\n"
    
    return formatted

# 주제별 프리셋
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

# 범용 AI 리서치 크루 클래스
class UnifiedResearchCrew:
    """통합된 AI 리서치 크루 시스템"""
    
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
        """에이전트 생성 (품질 모드에 따라 다름)"""
        
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
            backstory='''웹 검색을 통해 실시간 정보를 수집하고, 다양한 출처의 정보를 비판적으로 분석하여 
            신뢰할 수 있는 인사이트를 도출하는 숙련된 연구 전문가입니다.''',
            verbose=True,
            allow_delegation=False,
            tools=[improved_web_search_tool],
            llm=f"openai/{MODEL_NAME}",
            max_tokens=2000,
            temperature=0.7
        )

        # 품질 모드에 따른 작가 에이전트 생성
        if self.config.quality_mode == "korean_optimized":
            writer = self._create_korean_optimized_writer()
        else:
            writer = self._create_standard_writer()
        
        return planner, researcher, writer
    
    def _create_standard_writer(self):
        """표준 콘텐츠 작성 에이전트"""
        return Agent(
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
    
    def _create_korean_optimized_writer(self):
        """한국어 품질 강화된 콘텐츠 작성 에이전트"""
        return Agent(
            role='한국어 전문 작가',
            goal=f'{self.config.topic}에 대한 자연스럽고 정확한 한국어 블로그 작성',
            backstory='''한국어 전문 작가로서 복잡한 기술 내용을 
            자연스럽고 이해하기 쉬운 한국어로 표현합니다. 
            영어 표현을 사용하지 않고 순수 한국어만을 사용하며,
            정확한 정보와 실용적인 인사이트를 제공합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=f"openai/{MODEL_NAME}",
            max_tokens=2000,
            temperature=0.7  # 더 보수적인 온도
        )

    def create_tasks(self, planner, researcher, writer):
        """태스크 생성"""
        
        # 1. 검색 계획 수립
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
        
        # 2. 정보 수집
        research_task = Task(
            description=f'''이전 단계에서 생성된 검색 쿼리 목록을 활용하여 "{self.config.topic}"에 대한 심층 웹 검색을 수행합니다.

            **필수 수행 절차:**
            1. 이전 Task 결과에서 "SEARCH_QUERY_1:", "SEARCH_QUERY_2:" 등의 형식으로 된 검색 쿼리들을 찾아 추출합니다.
            2. 각 SEARCH_QUERY_X에서 따옴표 안의 검색어만 추출합니다.
            3. 추출된 **모든 검색어를 하나씩 순서대로** 'Web Search Tool'을 사용하여 검색합니다.
            4. 검색할 때마다 "🔍 검색 중: X/5 - [검색어]" 형태로 진행상황을 알려주세요.
            5. 만약 어떤 검색이 실패하거나 관련없는 결과가 나오면, 해당 주제의 대체 검색어를 만들어 다시 검색하세요.
            
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
            모든 생성된 검색 쿼리를 통해 얻은 최신 정보를 바탕으로 작성.''',
            
            agent=researcher,
            context=[planning_task]
        )

        # 3. 콘텐츠 작성 (품질 모드에 따라 다름)
        if self.config.quality_mode == "korean_optimized":
            write_task = self._create_korean_optimized_task(writer, research_task)
        else:
            write_task = self._create_standard_task(writer, research_task)
        
        return [planning_task, research_task, write_task]
    
    def _create_standard_task(self, writer, research_task):
        """표준 작성 태스크"""
        return Task(
            description=f'''연구 요약 보고서를 바탕으로 "{self.config.topic}"에 대한 
            **반드시 {self.config.language}로만 작성된** {self.config.report_type}을 작성합니다.
            
            **구체적 요구사항:**
            - 분량: {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어
            - 매력적인 {self.config.language} 제목과 부제목 사용
            - 명확하고 이해하기 쉬운 {self.config.language} 표현
            - 연구 보고서의 최신 정보를 반영한 현실적인 내용
            - 독자의 관심을 끄는 구성
            - 실제 사례나 구체적 예시 포함
            
            대상 독자: 해당 분야에 관심있는 일반인 및 전문가''',
            
            expected_output=f'''독자 친화적이고 정보가 풍부한 {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어 분량의 
            {self.config.report_type}. **완전히 {self.config.language}로만 작성**되었으며, 
            동적 웹 검색 결과를 반영한 현실적이고 유용한 내용 포함.''',
            
            agent=writer,
            context=[research_task]
        )
    
    def _create_korean_optimized_task(self, writer, research_task):
        """한국어 최적화 작성 태스크"""
        return Task(
            description=f'''
            수집된 연구 자료를 바탕으로 "{self.config.topic}"에 대한 자연스러운 한국어 블로그를 작성하세요.
            
            **절대 규칙:**
            1. 영어 단어, 문장, 표현을 절대 사용하지 마세요
            2. 모든 내용을 자연스러운 한국어로만 작성하세요
            3. 전문 용어도 한국어로 번역하여 사용하세요
            
            **글 구조:**
            1. 흥미로운 제목 (한국어만)
            2. 매력적인 도입부 (200-300자)
            3. 주요 내용 (5-6개 섹션):
               - 현재 동향과 발전 상황
               - 핵심 기술과 혁신 내용
               - 실제 활용 사례와 예시
               - 주요 기업과 시장 변화
               - 미래 전망과 예측
               - 결론과 시사점
            4. 마무리 요약 (150-200자)
            
            **작성 기준:**
            - 총 800-1000단어 (한국어 기준)
            - 구체적인 데이터와 사례 포함
            - 전문적이지만 일반인도 이해하기 쉬운 문체
            - 각 섹션에 적절한 소제목 사용
            - 실용적이고 유익한 정보 제공
            
            독자가 주제를 완전히 이해하고 실용적인 인사이트를 얻을 수 있도록 작성하세요.
            ''',
            agent=writer,
            expected_output=f"{self.config.topic}에 대한 고품질 순수 한국어 블로그 포스트 (800-1000단어)",
            context=[research_task]
        )
    
    def save_result(self, result):
        """결과를 파일로 저장"""
        if not result:
            logger.error("저장할 결과가 없습니다.")
            return False
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_report_{self.config.safe_topic}_{self.config.quality_mode}_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {self.config.topic} 연구 보고서\n\n")
                f.write(f"**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**품질 모드:** {self.config.quality_mode}\n")
                f.write(f"**언어:** {self.config.language}\n\n")
                f.write("---\n\n")
                f.write(str(result))
            
            logger.info(f"✅ 결과 저장 완료: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 파일 저장 실패: {e}")
            return False
    
    def research(self):
        """메인 리서치 실행 메서드"""
        try:
            logger.info(f"🚀 '{self.config.topic}' 연구 시작 (모드: {self.config.quality_mode})")
            
            # 검색 히스토리 초기화
            clear_search_history()
            
            # 에이전트 생성
            planner, researcher, writer = self.create_agents()
            
            # 태스크 생성
            tasks = self.create_tasks(planner, researcher, writer)
            
            # 크루 생성 및 실행
            crew = Crew(
                agents=[planner, researcher, writer],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # 결과 저장
            if self.save_result(result):
                logger.info(f"✅ '{self.config.topic}' 연구 완료")
                return result
            else:
                logger.error("❌ 결과 저장 실패")
                return None
                
        except Exception as e:
            logger.error(f"❌ 연구 실행 실패: {e}")
            return None

def main():
    """메인 실행 함수 - CLI 인터페이스 포함"""
    parser = argparse.ArgumentParser(description='통합 AI 리서치 크루 - 모든 주제에 대한 보고서 생성')
    parser.add_argument('--topic', '-t', 
                        default='2025년 최신 AI 트렌드', 
                        help='연구 주제 (또는 프리셋: ai, blockchain, health, etc.)')
    parser.add_argument('--queries', '-q', 
                        type=int, default=5, 
                        help='검색 쿼리 개수 (기본값: 5)')
    parser.add_argument('--words', '-w', 
                        default='700,900', 
                        help='단어 수 범위 (예: 700,900)')
    parser.add_argument('--type', '-r', 
                        default='블로그', 
                        help='보고서 유형 (블로그, 보고서, 분석서 등)')
    parser.add_argument('--language', '-l', 
                        default='한국어', 
                        help='출력 언어')
    parser.add_argument('--quality', '-m',
                        choices=['standard', 'korean_optimized'],
                        default='standard',
                        help='품질 모드 (standard: 표준, korean_optimized: 한국어 최적화)')
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
            raise ValueError()
    except:
        word_range = (700, 900)
        print("⚠️ 단어 수 형식 오류. 기본값 (700,900) 사용")
    
    # 프리셋 확인 및 주제 설정
    topic = get_preset_topic(args.topic)
    
    config = ResearchConfig(
        topic=topic,
        search_queries_count=args.queries,
        word_count_range=word_range,
        language=args.language,
        report_type=args.type,
        quality_mode=args.quality
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
    print(f"📝 목표 단어 수: {config.word_count_range[0]}-{config.word_count_range[1]}단어")
    print(f"🌟 품질 모드: {config.quality_mode}")
    
    # 리서치 실행
    crew = UnifiedResearchCrew(config)
    result = crew.research()
    
    if result:
        print(f"\n✅ '{config.topic}' 연구 보고서가 성공적으로 생성되었습니다!")
    else:
        print(f"\n❌ 작업 실패. 로그를 확인해보세요.")

def run_default():
    """기본 실행 함수 - CLI 없이 바로 동작"""
    print("🚀 통합 AI 리서치 크루 시작!")
    print("📋 기본 주제로 보고서를 생성합니다...")
    
    # 기본 설정으로 연구 실행
    config = ResearchConfig(
        topic="2025년 최신 AI 트렌드",
        search_queries_count=5,
        word_count_range=(700, 900),
        language="한국어",
        report_type="블로그",
        quality_mode="korean_optimized"  # 한국어 최적화 모드로 기본 설정
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
    print(f"📝 목표 단어 수: {config.word_count_range[0]}-{config.word_count_range[1]}단어")
    print(f"🌟 품질 모드: {config.quality_mode}")
    print("\n" + "="*60)
    
    # 리서치 실행
    crew = UnifiedResearchCrew(config)
    result = crew.research()
    
    if result:
        print(f"\n✅ '{config.topic}' 연구 보고서가 성공적으로 생성되었습니다!")
        print("💡 다른 주제로 연구하려면: python unified_research_crew.py --topic '원하는 주제'")
        print("💡 한국어 최적화 모드: python unified_research_crew.py --quality korean_optimized")
    else:
        print(f"\n❌ 작업 실패. 로그를 확인해보세요.")
    
    return result

if __name__ == "__main__":
    # 명령행 인자가 있으면 CLI 모드, 없으면 기본 실행
    if len(sys.argv) > 1:
        main()
    else:
        run_default()
