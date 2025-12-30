import streamlit as st
import pandas as pd
from utils import LLMCardExpert

# 페이지 기본 설정
st.set_page_config(page_title="혜택 시뮬레이션", layout="wide")
st.title("AI 상황별 혜택 시뮬레이션")
st.markdown("특정 소비 상황에서 **내 보유 카드**와 **새로운 추천 카드** 중 무엇이 유리한지 비교합니다.")

expert = LLMCardExpert()

# 사이드바 설정
with st.sidebar:
    st.header("로그인")
    if expert.users_df is not None:
        cust_ids = expert.users_df['CUST_ID'].astype(str).tolist()
        selected_user = st.selectbox("사용자 선택", cust_ids, index=0)
        
        my_inventory = expert.get_user_inventory(str(selected_user), source_type="batch1")
        st.caption(f"보유 카드: {len(my_inventory)}장")
        
        if my_inventory:
            with st.expander("보유 카드 목록"):
                for c in my_inventory:
                    st.text(f"- {expert.get_card_name(c)}")
    else:
        selected_user = None
        st.error("유저 데이터 로드 실패")

query = st.text_area("질문 입력", placeholder="예시: 이번 달 배달비가 30만원 나올 것 같은데 어떤 카드가 좋아?", height=100)

if st.button("내 카드 vs 추천 카드 비교 분석", type="primary"):
    if not query:
        st.warning("질문을 입력해주세요.")
    else:
        inventory = expert.get_user_inventory(selected_user, source_type="batch1") if selected_user else []
        
        # 분석 진행 및 상태 표시
        with st.status("AI 분석 중...", expanded=True) as status:
            intent, intent_usage = expert.analyze_intent(query)
            st.write(f"의도 파악 완료: {intent.categories} / {intent.amount:,}원")
            
            #
            results = expert.calculate_benefits(intent, user_inventory=inventory)
            status.update(label="비교 계산 완료", state="complete")
        
        col1, col2 = st.columns(2)
        
        # [왼쪽] 내 보유 카드 결과
        with col1:
            st.subheader("내 보유 카드 1위")
            if results["my_cards"]:
                df_my = pd.DataFrame(results["my_cards"])
                st.dataframe(
                    df_my,
                    column_config={
                        "card_name": "카드명",
                        "benefit_amt": st.column_config.NumberColumn("예상 혜택", format="%d원"),
                        "card_url": st.column_config.LinkColumn("링크", display_text="신청하기"),
                    },
                    column_order=["card_name", "benefit_amt", "card_url"], # [핵심] 보여줄 컬럼만 지정
                    width='stretch',
                    hide_index=True 
                )
            else:
                st.info("조건에 맞는 보유 카드가 없습니다.")

        # 시장 추천 카드 결과
        with col2:
            st.subheader("추천 카드 Top 5")
            if results["recs"]:
                df_recs = pd.DataFrame(results["recs"])
                st.dataframe(
                    df_recs,
                    column_config={
                        "card_name": "카드명",
                        "benefit_amt": st.column_config.NumberColumn("예상 혜택", format="%d원"),
                        "card_url": st.column_config.LinkColumn("링크", display_text="신청하기")
                    },
                    column_order=["card_name", "benefit_amt", "card_url"], # [핵심] 보여줄 컬럼만 지정
                    width='stretch',
                    hide_index=True 
                )
            else:
                st.error("추천 카드를 찾지 못했습니다.")

        st.divider()
        st.subheader(" AI 비교 분석 리포트")
        
        response_obj, url_map = expert.generate_simulation_report(query, intent, results)
        
        if response_obj:
            rows = []
            for item in response_obj.table_data:
                real_url = url_map.get(item.card_id, "#")
                
                # [방어 로직] 이름에 ID가 붙어있다면 제거
                clean_name = item.card_name
                if ":" in clean_name:
                    clean_name = clean_name.split(":")[-1].strip()
                    
                rows.append({
                    "구분": "⭐ BEST" if item.is_best else "",
                    "카드명": clean_name,  # 정제된 이름 사용
                    "주요 혜택": item.benefit_summary,
                    "계산 공식": item.benefit_formula, 
                    "연회비": item.annual_fee,
                    "예상 혜택": item.expected_benefit,
                    "신청": real_url
                })
            
            df = pd.DataFrame(rows)
            
            # 최종 테이블 출력
            st.dataframe(
                df,
                column_config={
                    "구분": st.column_config.TextColumn("추천", width="small"),
                    "카드명": st.column_config.TextColumn("카드 이름", width="medium"),
                    "주요 혜택": st.column_config.TextColumn("혜택 상세", width="large"),
                    "계산 공식": st.column_config.TextColumn("상세 계산 공식", width="large"),
                    "예상 혜택": st.column_config.NumberColumn("월 예상 혜택", format="%d원"),
                    "연회비": st.column_config.TextColumn("연회비", width="small"),
                    "신청": st.column_config.LinkColumn("링크", display_text="신청하기")
                },
                hide_index=True,
                width='stretch'
            )
            st.success(f"**AI 전문가 의견::** {response_obj.final_opinion}")
        else:
            st.warning("리포트를 생성하지 못했습니다.")