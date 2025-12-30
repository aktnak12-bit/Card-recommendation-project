import streamlit as st
import pandas as pd
from utils import LLMCardExpert

st.set_page_config(page_title="프리미엄 상담소", layout="wide")
st.title("AI 프리미엄 카드 컨설팅")
st.markdown("""
AI가 사용자의 소비 패턴을 심층 분석하여 **가장 적합한 핵심 후보군을 선별**하고, 정밀 시뮬레이션을 수행합니다.
""")

expert = LLMCardExpert()

with st.sidebar:
    st.header("로그인")
    if expert.users_df is not None:
        cust_ids = expert.users_df['CUST_ID'].astype(str).tolist()
        selected_user = st.selectbox("고객 ID", cust_ids, index=0)
        
        my_cards = expert.get_user_inventory(selected_user)
        st.caption(f"보유 카드: {len(my_cards)}장")
        
        with st.expander("보유 카드 목록"):
            for c in my_cards:
                st.text(f"- {expert.get_card_name(c)}")
    else:
        selected_user = None

st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(
        "상세한 소비 패턴을 알려주세요.", 
        placeholder="예시: 월 200만원 사용. 통신비 10만원, 관리비 20만원, 나머지는 배달과 스타벅스 위주입니다.",
        height=200
    )
with col2:
    st.info("**프리미엄 분석 팁**\n\n구체적인 수치를 입력할수록 정확합니다.\n- **고정 지출:** 관리비, 통신비\n- **변동 지출:** 식비, 쇼핑\n- **연회비 선호:** 3만원 이하 등")

if st.button("AI 정밀 분석 시작", type="primary"):
    if not user_input:
        st.warning("분석할 내용을 입력해주세요.")
    else:
        inventory = expert.get_user_inventory(selected_user) if selected_user else []
        
        with st.spinner("AI가 소비 패턴을 시뮬레이션하고 있습니다..."):
            response_obj, url_map = expert.recommend_by_prompt(user_input, user_inventory=inventory)
            
        if response_obj:
            st.divider()
            st.subheader(" 프리미엄 분석 리포트")
            
            # 1. 테이블 생성
            rows = []
            for item in response_obj.table_data:
                real_url = url_map.get(item.card_id, "#")
                rows.append({
                    "추천": "🥇 1위" if item.is_best else "",
                    "카드명": item.card_name,
                    "예상 혜택": item.expected_benefit,
                    "신청": real_url,
                    "연회비": item.annual_fee,
                    "혜택 요약": item.benefit_summary,
                    "계산 공식": item.benefit_formula
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                column_config={
                    "추천": st.column_config.TextColumn("순위", width="small"),
                    "카드명": st.column_config.TextColumn("카드명", width="medium"),
                    "예상 혜택": st.column_config.NumberColumn("월 예상 혜택", format="%d원"),
                    "신청": st.column_config.LinkColumn("링크", display_text="신청하기"),
                    "혜택 요약": st.column_config.TextColumn("혜택 상세", width="large"),
                    "계산 공식": st.column_config.TextColumn("상세 계산 공식", width="large")
                },
                hide_index=True,
                width="stretch"
            )
            
            # 2. 종합 의견
            st.success(f"**AI 전문가 의견::** {response_obj.final_opinion}")
        else:
            st.error("분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")