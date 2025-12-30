import streamlit as st
from utils import load_data

#페이지 기본 설정
st.set_page_config(page_title="AI 카드 추천 플랫폼", layout="wide")

# 메인 콘텐츠 영역
st.title("AI 카드 추천 플랫폼")
st.markdown("---")
st.info("왼쪽 사이드바에서 원하는 서비스를 선택하세요.")

# 데이터 로드 및 상태 표시
users, v2, batch1 = load_data()

col1, col2 = st.columns(2)
with col1:
    if users is not None: st.success(f"유저 데이터 로드됨 ({len(users)}명)")
    else: st.error("유저 데이터를 찾을 수 없습니다.")
with col2:
    if v2 and batch1: st.success("카드 데이터베이스 로드됨")
    else: st.error("카드 데이터베이스 일부 누락")