import streamlit as st
import pandas as pd
from utils import LLMCardExpert, display_consumption_summary

# [1] 페이지 기본 설정
st.set_page_config(page_title="개인 맞춤 카드 추천", layout="wide")

st.title("개인 맞춤 카드 추천 시스템")
st.markdown("""
과거 3개월 소비 내역을 분석하여 **현재 보유 카드의 적합성**을 점검하고, 
더 나은 혜택을 제공하는 **최적의 카드를 제안**합니다.
""")

# [2] 분석 전문가 객체 로드
expert = LLMCardExpert()

# [3] 사이드바: 고객 선택 및 인벤토리 확인
with st.sidebar:
    st.header(" 로그인")
    
    if expert.users_df is not None:
        cust_ids = expert.users_df['CUST_ID'].astype(str).tolist()
        # 데이터가 충분할 경우 기본 인덱스 설정
        default_index = 15 if len(cust_ids) > 15 else 0
        selected_id = st.selectbox("고객 ID", cust_ids, index=default_index)
        
        # 보유 카드 로드
        user_inventory = expert.get_user_inventory(str(selected_id))
        st.info(f"보유 카드: **{len(user_inventory)}장**")
        
        with st.expander("보유 카드 목록 보기"):
            if user_inventory:
                for card in user_inventory:
                    st.text(f"- {expert.get_card_name(card)}")
            else:
                st.text("보유 카드가 없습니다.")
    else:
        selected_id = None
        st.error("유저 데이터를 불러오지 못했습니다.")

# [4] 메인 영역: 소비 데이터 시각화 및 리포트 생성
if selected_id and expert.users_df is not None:
    target_user = expert.users_df[expert.users_df['CUST_ID'].astype(str) == selected_id].iloc[0]

    st.divider()

    # 소비 패턴 시각화 차트
    st.subheader(f" {selected_id} 고객님의 소비 패턴 분석")
    display_consumption_summary(target_user)

    # with st.expander("최근 3개월 소비 내역 상세 데이터 (RAW)", expanded=False):
    #     st.dataframe(target_user.to_frame(name="Value").T, use_container_width=True)

    # [AI 분석 실행]
    if st.button(" AI 맞춤 추천 리포트 생성", type="primary"):
        with st.spinner(f"Customer {selected_id}님의 소비 패턴을 심층 분석 중입니다..."):
            response_obj, url_map = expert.recommend_by_history(target_user, user_inventory=user_inventory)
        
        if response_obj:
            st.divider()
            st.subheader("AI 카드 추천 리포트")
            
            # 1. 데이터 가공
            rows = []
            for item in response_obj.table_data:
                real_url = url_map.get(item.card_id, "#")
                rows.append({
                    "구분": "⭐ BEST" if item.is_best else "추천",
                    "카드명": item.card_name,
                    "신청": real_url,
                    "월 예상 혜택": item.expected_benefit,
                    "연회비": item.annual_fee,
                    "주요 혜택": item.benefit_summary,
                    "계산 공식": item.benefit_formula, # 투명한 계산 로직 노출
                    "연회비": item.annual_fee
                })
            
            df = pd.DataFrame(rows)

            # 2. 결과 표 렌더링
            st.dataframe(
                df,
                column_config={
                    "구분": st.column_config.TextColumn("분류", width="small"),
                    "카드명": st.column_config.TextColumn("카드 이름", width="medium"),
                    "신청": st.column_config.LinkColumn("링크", display_text="신청하기"),
                    "월 예상 혜택": st.column_config.NumberColumn("예상 혜택", format="%d원"),
                    "연회비": st.column_config.TextColumn("연회비", width="small"),
                    "주요 혜택": st.column_config.TextColumn("혜택 상세", width="large"),
                    "계산 공식": st.column_config.TextColumn("상세 계산 근거", width="large")
                },
                hide_index=True,
                width='stretch'
            )

            # 3. AI 전문가 종합 의견
            st.info(f" **AI 전문가 의견:** \n{response_obj.final_opinion}")
        else:
            st.error("분석 결과를 생성하지 못했습니다.")