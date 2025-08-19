"""
한국어 품질 향상을 위한 개선된 프롬프트
"""
import os
from improved_research_crew import ImprovedResearchCrew

class KoreanOptimizedResearchCrew(ImprovedResearchCrew):
    """한국어 품질에 최적화된 리서치 크루"""
    
    def create_writer(self):
        """한국어 품질 강화된 콘텐츠 작성 에이전트"""
        from crewai import Agent
        
        return Agent(
            role='한국어 전문 작가',
            goal=f'{self.topic}에 대한 자연스럽고 정확한 한국어 블로그 작성',
            backstory='''한국어 전문 작가로서 복잡한 기술 내용을 
            자연스럽고 이해하기 쉬운 한국어로 표현합니다. 
            영어 표현을 사용하지 않고 순수 한국어만을 사용하며,
            정확한 정보와 실용적인 인사이트를 제공합니다.''',
            verbose=True,
            allow_delegation=False,
            llm=self.llm_config,
            max_tokens=2000,
            temperature=0.7  # 더 보수적인 온도
        )

    def create_tasks(self, planner, researcher, writer):
        """한국어 품질 강화된 태스크"""
        from crewai import Task
        
        # 기존 계획과 리서치 태스크 유지
        tasks = super().create_tasks(planner, researcher, writer)
        
        # 작성 태스크만 수정
        writing_task = Task(
            description=f'''
            수집된 연구 자료를 바탕으로 "{self.topic}"에 대한 자연스러운 한국어 블로그를 작성하세요.
            
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
            
            **품질 검증:**
            - 작성 완료 후 영어 표현이 없는지 재확인
            - 문법과 맞춤법 정확성 확인
            - 내용의 논리적 흐름 확인
            
            독자가 주제를 완전히 이해하고 실용적인 인사이트를 얻을 수 있도록 작성하세요.
            ''',
            agent=writer,
            expected_output=f"{self.topic}에 대한 고품질 순수 한국어 블로그 포스트 (800-1000단어)"
        )
        
        # 마지막 태스크만 교체
        tasks[-1] = writing_task
        return tasks

def main():
    """한국어 최적화 테스트"""
    print("🇰🇷 한국어 품질 강화 AI 리서치 크루")
    print("=" * 50)
    
    topic = input("📝 연구 주제를 입력하세요: ").strip()
    if not topic:
        topic = "인공지능 최신 동향"
        print(f"기본 주제 사용: {topic}")
    
    crew = KoreanOptimizedResearchCrew(topic)
    result = crew.run_research()
    
    print("\n" + "=" * 50)
    print("📋 최종 결과 (한국어 최적화):")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
