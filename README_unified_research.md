# 통합 AI 리서치 크루 사용 가이드

## 📚 기본 사용법

### 1. 기본 실행 (인자 없이)
```bash
python3 unified_research_crew.py
```
- 기본 주제: "2025년 최신 AI 트렌드"
- 품질 모드: korean_optimized (한국어 최적화)

### 2. 프리셋 주제 사용
```bash
# 프리셋 목록 확인
python3 unified_research_crew.py --list-presets

# 프리셋 사용
python3 unified_research_crew.py --topic blockchain --quality korean_optimized
python3 unified_research_crew.py --topic ai --quality standard
```

### 3. 커스텀 주제
```bash
python3 unified_research_crew.py --topic "2025년 인공지능 동향" --quality korean_optimized --queries 3
```

### 4. 전체 옵션
```bash
python3 unified_research_crew.py \
  --topic "원하는 주제" \
  --quality korean_optimized \
  --queries 5 \
  --words 800,1200 \
  --type "보고서" \
  --language "한국어"
```

## 🎛️ 옵션 설명

- `--topic, -t`: 연구 주제 또는 프리셋명
- `--quality, -m`: 품질 모드 (standard/korean_optimized)
- `--queries, -q`: 검색 쿼리 개수 (기본: 5)
- `--words, -w`: 단어 수 범위 (기본: 700,900)
- `--type, -r`: 보고서 유형 (기본: 블로그)
- `--language, -l`: 출력 언어 (기본: 한국어)

## 📦 프리셋 주제 목록

- `ai`: 2025년 최신 AI 트렌드
- `blockchain`: 2025년 블록체인 기술 발전
- `climate`: 지속가능한 기후 기술 혁신
- `health`: 디지털 헬스케어 기술 트렌드
- `fintech`: 핀테크 산업 최신 동향
- `architecture`: 현대 건축 기술 혁신
- `education`: 교육 기술 디지털 전환
- `energy`: 재생 에너지 기술 발전
- `space`: 우주 기술 및 탐사 동향
- `food`: 푸드테크 산업 혁신

## 🔧 품질 모드 비교

### Standard 모드
- 일반적인 콘텐츠 작성
- 다양한 언어 지원
- 균형잡힌 온도 설정

### Korean Optimized 모드
- 순수 한국어만 사용
- 영어 표현 완전 배제
- 한국어 문법에 최적화
- 더 보수적인 온도 설정

## 📄 출력 파일

생성된 보고서는 다음 형식으로 저장됩니다:
```
research_report_{주제}_{품질모드}_{타임스탬프}.md
```

예시:
```
research_report_2025년_블록체인_기술_발전_korean_optimized_20250819_180914.md
```

## 🚀 빠른 시작 예시

```bash
# 1. 블록체인 보고서 (한국어 최적화)
python3 unified_research_crew.py --topic blockchain --quality korean_optimized

# 2. AI 트렌드 보고서 (표준 모드)
python3 unified_research_crew.py --topic ai --quality standard

# 3. 커스텀 주제 (빠른 테스트용 - 쿼리 3개)
python3 unified_research_crew.py --topic "메타버스 기술" --queries 3
```
