"""
개선된 AI 리서치 크루 - 명확한 지시사항과 안정성 강화
"""
import os
import logging
from datetime import datetime
import litellm
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from fixed_search_tool import improved_web_search_tool, clear_search_history

# 환경 설정
load_dotenv()
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedResearchCrew:
    """개선된 AI 리서치 크루 - 명확한 태스크 분할과 에러 처리"""
    
    def __init__(self, topic: str, language: str = "한국어"):
        self.topic = topic
        self.language = language
        self.setup_llm()
        
    def setup_llm(self):
        """LLM 설정"""
        model_name = os.getenv("DEFAULT_LLM", "mistral-small3.2")
        api_base = os.getenv("DEFAULT_URL", "http://192.168.100.26:11434")
        api_key = os.getenv("DEFAULT_API_KEY", "ollama")
        
        if not api_base.endswith('/v1'):
            api_base = api_base.rstrip('/') + '/v1'
            
        self.llm_config = f"openai/{model_name}"
        
        litellm.api_base = api_base
        litellm.api_key = api_key
        litellm.drop_params = True
        
        logger.info(f"LLM 설정 완료: {model_name} @ {api_base}")

    def create_search_planner(self) -> Agent:
        """검색 계획 수립 에이전트"""
        return Agent(
            role='검색 쿼리 전략가',
            goal=f'{self.topic}에 대한 다양하고 효과적인 검색 쿼리 5개 생성',
            backstory='''검색 전략 전문가로서, 주제를 다각도로 분석하여 
            서로 다른 관점의 검색 쿼리를 생성합니다. 중복을 피하고 
            포괄적인 정보 수집이 가능한 쿼리를 설계합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=self.llm_config,
            max_tokens=800,
            temperature=0.7
        )

    def create_researcher(self) -> Agent:
        """리서치 수행 에이전트"""
        return Agent(
            role='정보 수집 전문가',
            goal=f'{self.topic}에 대한 신뢰할 수 있는 최신 정보 수집',
            backstory='''웹 검색 도구를 사용하여 체계적으로 정보를 수집하고,
            수집된 정보의 신뢰성과 관련성을 평가합니다. 
            각 검색마다 서로 다른 키워드를 사용하여 중복을 방지합니다.''',
            verbose=True,
            allow_delegation=False,
            tools=[improved_web_search_tool],
            llm=self.llm_config,
            max_tokens=1500,
            temperature=0.6
        )

    def create_writer(self) -> Agent:
        """콘텐츠 작성 에이전트"""
        return Agent(
            role='전문 콘텐츠 작가',
            goal=f'{self.topic}에 대한 고품질 {self.language} 블로그 포스트 작성',
            backstory=f'''전문적이면서도 이해하기 쉬운 {self.language} 콘텐츠를 작성합니다.
            복잡한 기술적 내용을 일반 독자가 이해할 수 있도록 명확하게 설명하며,
            실용적인 인사이트와 구체적인 예시를 포함합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=self.llm_config,
            max_tokens=2000,
            temperature=0.8
        )

    def create_tasks(self, planner: Agent, researcher: Agent, writer: Agent) -> list:
        """개선된 태스크 생성"""
        
        # 1단계: 검색 계획 수립
        planning_task = Task(
            description=f'''
            주제: "{self.topic}"
            
            이 주제에 대해 포괄적인 리서치를 위한 5개의 서로 다른 영어 검색 쿼리를 생성하세요.
            
            **필수 요구사항:**
            1. 각 쿼리는 완전히 다른 키워드를 사용해야 합니다
            2. 다음 관점들을 포함해야 합니다:
               - 최신 동향 (latest trends, recent developments)
               - 전문가 분석 (expert analysis, research findings)
               - 실제 사례 (case studies, real-world applications)
               - 미래 전망 (future outlook, predictions)
               - 산업 영향 (industry impact, market analysis)
            
            **출력 형식 (반드시 준수):**
            QUERY_1: "첫 번째 검색 쿼리"
            QUERY_2: "두 번째 검색 쿼리"  
            QUERY_3: "세 번째 검색 쿼리"
            QUERY_4: "네 번째 검색 쿼리"
            QUERY_5: "다섯 번째 검색 쿼리"
            
            각 쿼리는 3-8개 단어로 구성하고, 구체적이고 검색 가능한 용어를 사용하세요.
            ''',
            agent=planner,
            expected_output="5개의 서로 다른 영어 검색 쿼리 목록"
        )

        # 2단계: 정보 수집
        research_task = Task(
            description=f'''
            검색 계획을 바탕으로 "{self.topic}"에 대한 정보를 수집하세요.
            
            **수행 방법:**
            1. 제공받은 5개 검색 쿼리를 각각 한 번씩만 사용하여 웹 검색을 수행하세요
            2. 각 검색마다 다른 검색어를 사용하여 중복을 피하세요
            3. 검색 결과에서 핵심 정보를 추출하세요
            
            **중요 사항:**
            - 같은 검색어를 반복 사용하지 마세요
            - 검색 오류가 발생하면 비슷하지만 다른 키워드로 시도하세요
            - 각 검색 결과에서 핵심 내용을 요약하세요
            
            **수집할 정보:**
            - 최신 동향과 발전사항
            - 주요 통계 및 데이터
            - 전문가 의견 및 분석
            - 구체적 사례 및 적용 분야
            - 미래 전망 및 예측
            
            모든 정보를 종합하여 포괄적인 연구 보고서를 작성하세요.
            ''',
            agent=researcher,
            expected_output=f"{self.topic}에 대한 종합적인 연구 자료 및 핵심 인사이트"
        )

        # 3단계: 콘텐츠 작성
        writing_task = Task(
            description=f'''
            수집된 연구 자료를 바탕으로 "{self.topic}"에 대한 고품질 블로그 포스트를 작성하세요.
            
            **글 구조:**
            1. 매력적인 제목
            2. 흥미로운 도입부 (200-300자)
            3. 주요 내용 (5-6개 섹션)
               - 현재 상황 및 최신 동향
               - 핵심 기술 및 혁신사항
               - 실제 적용 사례
               - 주요 기업 및 시장 동향
               - 미래 전망 및 예측
               - 시사점 및 결론
            4. 마무리 및 요약 (150-200자)
            
            **작성 요구사항:**
            - 총 800-1000단어
            - 100% {self.language}로 작성
            - 구체적인 예시와 수치 포함
            - 전문적이지만 이해하기 쉬운 문체
            - 각 섹션에 소제목 사용
            - 실용적인 인사이트 제공
            
            **절대 금지사항:**
            - 영어 단어나 문장 사용 금지
            - 모호하거나 추상적인 표현 금지
            - 검증되지 않은 정보 포함 금지
            
            독자가 주제에 대해 명확히 이해할 수 있도록 상세하고 유익한 내용을 작성하세요.
            ''',
            agent=writer,
            expected_output=f"{self.topic}에 대한 고품질 {self.language} 블로그 포스트 (800-1000단어)"
        )

        return [planning_task, research_task, writing_task]

    def run_research(self) -> str:
        """리서치 실행"""
        try:
            # 검색 히스토리 초기화
            clear_search_history()
            
            # 에이전트 생성
            planner = self.create_search_planner()
            researcher = self.create_researcher()
            writer = self.create_writer()
            
            # 태스크 생성
            tasks = self.create_tasks(planner, researcher, writer)
            
            # 크루 생성 및 실행
            crew = Crew(
                agents=[planner, researcher, writer],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                max_execution_time=900  # 15분 제한
            )
            
            logger.info(f"🚀 '{self.topic}' 리서치 시작")
            result = crew.kickoff()
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improved_research_report_{self.topic.replace(' ', '_')}_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {self.topic} - 리서치 보고서\n\n")
                f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(str(result))
            
            logger.info(f"✅ 리서치 완료. 결과 저장: {filename}")
            return str(result)
            
        except Exception as e:
            logger.error(f"❌ 리서치 실행 중 오류: {e}")
            return f"리서치 실행 중 오류가 발생했습니다: {str(e)}"

def main():
    """메인 실행 함수"""
    print("🔬 개선된 AI 리서치 크루")
    print("=" * 50)
    
    topic = input("📝 연구 주제를 입력하세요: ").strip()
    if not topic:
        topic = "2025년 최신 AI 트렌드"
        print(f"기본 주제 사용: {topic}")
    
    crew = ImprovedResearchCrew(topic)
    result = crew.run_research()
    
    print("\n" + "=" * 50)
    print("📋 최종 결과:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
