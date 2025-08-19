"""
개선된 웹 검색 도구 - 중복 쿼리 방지 및 에러 처리 강화
"""
import os
import logging
from datetime import datetime
import hashlib
from typing import Set, List, Dict, Any
from crewai.tools import tool

# DuckDuckGo Search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# 전역 변수로 검색 히스토리 관리
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
        # 캐시된 결과가 있으면 반환
        if query_hash in _search_results_cache:
            return f"🔄 (캐시됨) {_search_results_cache[query_hash]}"
        else:
            return f"⚠️ 이미 검색한 쿼리입니다: '{query}'. 다른 검색어를 시도해보세요."
    
    # 검색 히스토리에 추가
    _search_history.add(query_hash)
    
    try:
        logging.info(f"🔍 웹 검색 시작: '{query}'")
        
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
            logging.warning(error_msg)
            return error_msg
        
        # 결과 포매팅
        formatted_results = format_search_results(query, results)
        
        # 캐시에 저장
        _search_results_cache[query_hash] = formatted_results
        
        logging.info(f"✅ 검색 완료: {len(results)}개 결과")
        return formatted_results
        
    except Exception as e:
        error_msg = f"❌ 검색 오류 발생: {str(e)}"
        logging.error(f"웹 검색 오류 - Query: '{query}', Error: {e}")
        
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

# 검색 상태 확인 유틸리티
def get_search_stats() -> Dict[str, Any]:
    """현재 검색 상태 반환"""
    return {
        "searches_performed": len(_search_history),
        "cached_results": len(_search_results_cache),
        "search_hashes": list(_search_history)
    }
