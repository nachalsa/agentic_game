import os
import logging
from datetime import datetime
import litellm
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import argparse
import re

# DuckDuckGo Search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

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
                 word_count_range: tuple = (700, 900),
                 language: str = "한국어",
                 report_type: str = "블로그"):
        self.topic = topic
        self.search_queries_count = search_queries_count
        self.word_count_range = word_count_range
        self.language = language
        self.report_type = report_type
        
        # 파일명용 안전한 토픽명 생성
        self.safe_topic = re.sub(r'[^\w\s-]', '', topic.replace(' ', '_'))[:50]

# 환경 변수
MODEL_NAME = os.getenv("DEFAULT_LLM", "cpatonn/Devstral-Small-2507-AWQ")
API_BASE_URL = os.getenv("DEFAULT_URL", "http://localhost:54321")
API_KEY = os.getenv("DEFAULT_API_KEY", "huntr/x_How_It's_Done")
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "600"))

if not API_BASE_URL.endswith('/v1'):
    API_BASE_URL = API_BASE_URL.rstrip('/') + '/v1'

# 웹 검색 도구
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """웹에서 정보를 검색하는 도구"""
    try:
        logger.info(f"웹 검색: '{query}'")
        ddgs = DDGS()
        results = ddgs.text(query=query, region='wt-wt', safesearch='moderate', max_results=5)
        
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
        
        logger.info(f"검색 완료: {len(results)}개 결과")
        return formatted_results
        
    except Exception as e:
        error_msg = f"검색 오류: {str(e)}"
        logger.error(error_msg)
        return error_msg

# 주제별 프리셋 (선택사항)
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
        """범용 에이전트 생성"""
        
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
        """범용 태스크 생성"""
        
        # 1. 검색 계획 수립
        planning_task = Task(
            description=f'''"{self.config.topic}"에 대한 포괄적인 연구를 수행해야 합니다.
            
            이 주제를 다음 관점에서 분석하여 {self.config.search_queries_count}개의 구체적이고 효과적인 영어 웹 검색 쿼리를 생성하세요:
            
            1. 최신 동향 및 발전사항 (latest trends, developments, innovations)
            2. 전문가 분석 및 연구 결과 (expert analysis, research, studies)  
            3. 실제 적용 사례 및 케이스 스터디 (case studies, applications, examples)
            4. 미래 전망 및 예측 (future outlook, predictions, forecasts)
            5. 산업별 영향 및 활용 (industry impact, implementation)
            
            각 쿼리는 명확하고 독립적으로 검색 가능해야 하며, 2024-2025년 최신 정보를 얻을 수 있도록 구성하세요.''',
            
            expected_output=f'''{self.config.search_queries_count}개의 영어 검색 쿼리 목록.
            각 쿼리는 한 줄로 명확하게 구분되어야 합니다. (예: `- "query"` 형식)
            주제: {self.config.topic}에 최적화된 검색어들''',
            agent=planner
        )
        
        # 2. 정보 수집
        research_task = Task(
            description=f'''이전 단계에서 생성된 검색 쿼리 목록을 활용하여 "{self.config.topic}"에 대한 심층 웹 검색을 수행합니다.

            **수행 절차:**
            1. 이전 Task 결과에서 제공된 **각각의 검색 쿼리**를 정확히 추출합니다.
            2. 추출된 **모든 쿼리에 대해** 'Web Search Tool'을 **순서대로 개별적으로 사용**합니다.
            3. 각 검색 결과를 분석하고 다음 쿼리로 진행합니다.
            4. 모든 검색 완료 후, 수집된 **모든 정보**를 종합 분석하여 보고서를 작성합니다.
            
            보고서는 다음을 포함해야 합니다:
            - 주요 트렌드 및 발전사항
            - 핵심 통계 및 데이터
            - 구체적 사례 및 실무 적용
            - 전문가 의견 및 분석
            - 미래 전망
            
            **중요:** 검색 결과가 영어로 나와도 보고서는 {self.config.language}로 작성해야 합니다.''',
            
            expected_output=f'''"{self.config.topic}"에 대한 주요 인사이트, 최신 통계 및 실제 예시를 포함하는 
            400-500단어 분량의 상세한 연구 요약 보고서 ({self.config.language}).
            모든 생성된 검색 쿼리를 통해 얻은 최신 정보를 바탕으로 작성.''',
            
            agent=researcher,
            context=[planning_task]
        )

        # 3. 콘텐츠 작성
        write_task = Task(
            description=f'''연구 요약 보고서를 바탕으로 "{self.config.topic}"에 대한 
            {self.config.language} {self.config.report_type}을 작성합니다.
            
            **요구사항:**
            - 분량: {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어
            - 언어: {self.config.language}
            - 매력적인 제목과 부제목 사용
            - 명확하고 이해하기 쉬운 언어
            - 연구 보고서의 최신 정보를 반영한 현실적인 내용
            - 독자의 관심을 끄는 구성
            - 실제 사례나 구체적 예시 포함
            
            **구조:**
            1. 흥미로운 도입부
            2. 주요 내용 (연구 결과 기반)
            3. 실생활/산업 적용 사례
            4. 미래 전망
            5. 결론 및 요약
            
            대상 독자: 해당 분야에 관심있는 일반인 및 전문가''',
            
            expected_output=f'''독자 친화적이고 정보가 풍부한 {self.config.word_count_range[0]}-{self.config.word_count_range[1]}단어 분량의 
            {self.config.report_type}. {self.config.language}로 작성되었으며, 
            동적 웹 검색 결과를 반영한 현실적이고 유용한 내용 포함.''',
            
            agent=writer,
            context=[research_task]
        )
        
        return planning_task, research_task, write_task
    
    def save_result(self, result):
        """결과를 파일로 저장"""
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
                f.write(f"검색 쿼리 수: {self.config.search_queries_count}개\n\n")
                f.write("---\n\n")
                f.write(str(result))
            
            logger.info(f"결과가 {filename}에 저장되었습니다.")
            return filename
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return None
    
    def research(self):
        """메인 리서치 실행 메서드"""
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 범용 AI 리서치 크루 시작")
            logger.info(f"📋 주제: {self.config.topic}")
            logger.info(f"📊 보고서 유형: {self.config.report_type}")
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
            logger.info("예상 소요 시간: 3-5분")
            
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
    """메인 실행 함수 - CLI 인터페이스 포함"""
    parser = argparse.ArgumentParser(description='범용 AI 리서치 크루 - 모든 주제에 대한 보고서 생성')
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
        word_count_range=word_range,
        language=args.language,
        report_type=args.type
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
    print(f"📝 목표 단어 수: {config.word_count_range[0]}-{config.word_count_range[1]}단어")
    
    # 리서치 실행
    crew = UniversalResearchCrew(config)
    result = crew.research()
    
    if result:
        print(f"\n✅ '{config.topic}' 연구 보고서가 성공적으로 생성되었습니다!")
    else:
        print(f"\n❌ 작업 실패. 로그를 확인해보세요.")

def run_default():
    """기본 실행 함수 - CLI 없이 바로 동작"""
    print("🚀 범용 AI 리서치 크루 시작!")
    print("📋 기본 주제로 보고서를 생성합니다...")
    
    # 기본 설정으로 연구 실행
    config = ResearchConfig(
        topic="2025년 최신 AI 트렌드",
        search_queries_count=5,
        word_count_range=(700, 900),
        language="한국어",
        report_type="블로그"
    )
    
    print(f"🎯 연구 주제: {config.topic}")
    print(f"📊 보고서 유형: {config.report_type}")
    print(f"🔍 검색 쿼리: {config.search_queries_count}개")
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
    # 패키지 확인
    try:
        import duckduckgo_search
        logger.info("✅ duckduckgo-search 패키지 확인됨")
    except ImportError:
        logger.error("❌ duckduckgo-search 패키지가 필요합니다")
        print("💡 설치 명령: pip install duckduckgo-search")
        exit(1)
    
    # 명령행 인자가 있으면 CLI 모드, 없으면 기본 실행
    import sys
    if len(sys.argv) > 1:
        main()  # CLI 모드
    else:
        run_default()  # 기본 실행 모드