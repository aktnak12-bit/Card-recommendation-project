import streamlit as st
import plotly.express as px
import pandas as pd
import json
import os
import numpy as np
import urllib.parse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

# 라이브러리 체크
try:
    from rank_bm25 import BM25Okapi
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("필수 라이브러리가 없습니다. 터미널에서 `pip install rank_bm25 scikit-learn numpy`를 실행하세요.")

load_dotenv()
# 1. 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

USER_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data_sample_10.parquet")
MINIFIED_DATA_PATH = os.path.join(DATA_DIR, "naver_card_minified.json")      # Master Data
CARD_BATCH1_PATH = os.path.join(DATA_DIR, "naver_card_optimized_batch1.json") # Calculation Data
CARD_V2_PATH = os.path.join(DATA_DIR, "naver_card_calculable_v2.json")      # Fallback

# 2. 데이터 구조 정의 (Pydantic Schema)
class UserIntent(BaseModel):
    categories: List[str] = Field(description="관련된 소비 카테고리 (예: oil, shopping)")
    amount: int = Field(default=500000, description="결제 예상 금액")
    target_merchant: Optional[str] = Field(default=None, description="특정 가맹점 이름")
    reasoning: str = Field(description="이 의도를 추출하게 된 논리적 근거")

class CardAnalysisItem(BaseModel):
    card_id: str = Field(description="카드 식별자 (예: MY_0, NEW_1)")
    card_name: str = Field(description="카드 이름 (예: 신한카드 Mr.Life)")
    benefit_summary: str = Field(description="주요 혜택 요약 (예: 주유 10% 할인)")
    benefit_formula: str = Field(description="상세 혜택 계산 공식 (예: 주유 400,000원 * 10% = 40,000원)")
    annual_fee: str = Field(description="연회비 (예: 15,000원)")
    expected_benefit: int = Field(description="월 예상 총 혜택 금액 (숫자만, 예: 35000)")
    is_best: bool = Field(description="가장 추천하는 카드 여부 (True/False)")

class CardAnalysisResponse(BaseModel):
    table_data: List[CardAnalysisItem] = Field(description="비교 분석표 데이터 리스트")
    final_opinion: str = Field(description="종합 추천 의견 (3줄 이내 요약)")

# 3. 데이터 로드 함수
@st.cache_data
def load_data():
    users_df = None
    minified_data = []    
    batch1_data = [] 
    
    if os.path.exists(USER_DATA_PATH):
        try: users_df = pd.read_parquet(USER_DATA_PATH, engine='pyarrow')
        except: pass

    if os.path.exists(MINIFIED_DATA_PATH):
        try:
            with open(MINIFIED_DATA_PATH, "r", encoding="utf-8") as f:
                minified_data = json.load(f)
        except: pass
    elif os.path.exists(CARD_V2_PATH):
        try:
            with open(CARD_V2_PATH, "r", encoding="utf-8") as f:
                minified_data = json.load(f) 
        except: pass

    if os.path.exists(CARD_BATCH1_PATH):
        try:
            with open(CARD_BATCH1_PATH, "r", encoding="utf-8") as f:
                batch1_data = json.load(f)
        except: pass
            
    return users_df, minified_data, batch1_data


# 4. 하이브리드 검색기 (RRF 방식 유지 - 성능이 좋으므로 공통 사용)
class HybridRetriever:
    def __init__(self, card_data: List[dict], client: genai.Client):
        self.client = client
        self.card_data = card_data
        self.documents = []
        
        for c in card_data:
            name = c.get('t') or c.get('title') or c.get('n') or c.get('card_name') or "Unknown"
            benefits = c.get('b') or c.get('benefits') or []
            if isinstance(benefits, list):
                ben_text = " ".join([str(b.get('d') or b.get('desc') or "") for b in benefits])
            else:
                ben_text = str(benefits)
            self.documents.append(f"{name} {ben_text}")

        tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.embeddings = self._generate_embeddings(self.documents)

    def _generate_embeddings(self, texts: List[str]):
        embeddings = []
        batch_size = 50
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch
                )
                batch_embeddings = [e.values for e in response.embeddings]
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
        except Exception as e:
            print(f"Embedding Error: {e}")
            return np.zeros((len(texts), 768)) 

    def search(self, query: str, top_k: int = 40, k: int = 60):
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1] 

        try:
            query_resp = self.client.models.embed_content(
                model="text-embedding-004",
                contents=query
            )
            query_embedding = np.array(query_resp.embeddings[0].values).reshape(1, -1)
            vector_scores = cosine_similarity(query_embedding, self.embeddings)[0]
            vector_ranks = np.argsort(vector_scores)[::-1]
        except Exception as e:
            vector_ranks = np.arange(len(self.card_data))

        rrf_scores = np.zeros(len(self.card_data))
        for rank, doc_idx in enumerate(bm25_ranks): rrf_scores[doc_idx] += 1 / (k + rank)
        for rank, doc_idx in enumerate(vector_ranks): rrf_scores[doc_idx] += 1 / (k + rank)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        return [self.card_data[i] for i in top_indices]

# 소비 분석 차트 (공통)
def display_consumption_summary(target_user):
    import pandas as pd
    pd.set_option('future.no_silent_downcasting', True)

    def get_sum(cols):
        valid_cols = [c for c in cols if c in target_user.index]
        if not valid_cols: return 0
        return target_user[valid_cols].fillna(0).infer_objects(copy=False).sum()

    summary = {
        "식비/생활": get_sum(['R3M_FOOD_AMT', 'R3M_SS_AMT', 'R3M_CONV_AMT', 'R3M_DLV_AMT', 'R3M_STARBUCKS_AMT']),
        "쇼핑/마트": float(target_user.filter(like='_DEP_').sum() + target_user.filter(like='_MART_').sum() + target_user.get('R3M_E_COMM_AMT', 0)),
        "여가/문화": get_sum(['R3M_ENT_AMT', 'R3M_CUL_AMT', 'R3M_ACCO_AMT', 'R3M_TRAVEL_AMT']),
        "교통/주유": get_sum(['R3M_TRANS_AMT', 'R3M_OIL_AMT', 'R3M_E_CHARGE_AMT']),
        "의료/교육": get_sum(['R3M_EDU_AMT', 'R3M_MED_AMT', 'R3M_BEAUTY_AMT'])
    }
    
    df_plot = pd.DataFrame([{"범주": k, "금액": v} for k, v in summary.items() if v > 0])
    if df_plot.empty:
        st.warning("분석할 소비 이력이 존재하지 않는 유저입니다.")
        return
    
    fig = px.pie(df_plot, values='금액', names='범주', hole=0.5, title="최근 3개월 소비 패턴", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textinfo='percent+label', hovertemplate="<b>%{label}</b><br>지출액: %{value:,.0f}천원<br>비중: %{percent}")
    st.plotly_chart(fig, width='stretch')

# 5. 통합 AI 전문가 클래스
class LLMCardExpert:
    def __init__(self):
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            st.error("Google API Key가 없습니다.")
            return

        self.client = genai.Client(api_key=api_key)
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model_name = "gemini-3-flash-preview" 
        
        self.users_df, self.minified_db, self.batch1_db = load_data()
        self.batch1_lookup = self._build_batch1_lookup(self.batch1_db)
        
        self.retriever_minified = None
        if self.minified_db and self.client:
            self.retriever_minified = HybridRetriever(self.minified_db, self.client)
            
        self.retriever_batch1 = None
        if self.batch1_db and self.client:
            self.retriever_batch1 = HybridRetriever(self.batch1_db, self.client)

    def _normalize_name(self, name: str):
        if not name: return ""
        return name.replace("네이버 신용카드 정보:", "").replace(" ", "").lower()

    def _build_batch1_lookup(self, batch1_data):
        lookup = {}
        for card in batch1_data:
            raw_name = card.get('card_name') or card.get('n') or ""
            norm_name = self._normalize_name(raw_name)
            if norm_name: lookup[norm_name] = card
        return lookup

    def get_card_name(self, card: dict) -> str:
        if not card: return "데이터 없음"
        return (card.get('card_name') or card.get('t') or card.get('n') or card.get('title') or "이름 없음")

    def _generate_url(self, card):
        possible_keys = ['u', 'url']
        for key in possible_keys:
            if card.get(key): return card.get(key)
        card_name = self.get_card_name(card)
        if card_name and card_name != "이름 없음":
            encoded = urllib.parse.quote(card_name)
            return f"https://search.naver.com/search.naver?query={encoded}"
        return "https://m.naver.com"

    def _detach_url_from_card(self, card: dict, unique_id: str) -> str:
        final_url = self._generate_url(card)
        card_copy = card.copy()
        card_copy['id'] = unique_id
        for k in ['u', 'url']: card_copy.pop(k, None)
        return final_url, card_copy

    def get_user_inventory(self, user_id: str, source_type: str = "minified"):
        if self.users_df is None or 'user_ib' not in self.users_df.columns: return []
        try:
            user_row = self.users_df[self.users_df['CUST_ID'].astype(str) == str(user_id)]
            if user_row.empty: return []
            ib_str = user_row.iloc[0]['user_ib']
            if not ib_str: return []
            indices = [int(idx) for idx in ib_str.split(',') if idx.strip().isdigit()]
            my_minified_cards = [self.minified_db[i] for i in indices if 0 <= i < len(self.minified_db)]

            if source_type == "minified": return [c.copy() for c in my_minified_cards]
            elif source_type == "batch1":
                my_batch1_cards = []
                for m_card in my_minified_cards:
                    m_name = m_card.get('t') or m_card.get('n') or m_card.get('card_name') or ""
                    norm_name = self._normalize_name(m_name)
                    if norm_name in self.batch1_lookup:
                        my_batch1_cards.append(self.batch1_lookup[norm_name].copy())
                return my_batch1_cards
        except Exception: return []

    def analyze_intent(self, query: str):
        prompt = f"""다음 질문을 분석하여 의도를 추출하세요: '{query}'
        당신은 카드 혜택 계산을 위한 의도 분석기입니다.
        사용자의 질문에서 다음 정보를 추출하세요:
        1. target_merchant: 언급된 특정 상점 이름 (예: 쿠팡, 스타벅스)
        2. categories: 관련 소비 분야 (예: shopping, cafe, food, oil)
        3. amount: 결제 예정 금액 (숫자만 추출, 언급 없으면 500000 기본값)
        
        [추출 예시]
        - "쿠팡에서 10만원 살거야" -> merchant: "쿠팡", categories: ["shopping"], amount: 100000
        """
        try:
            res = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=UserIntent)
            )
            usage = res.usage_metadata
            return res.parsed, {"input": usage.prompt_token_count, "output": usage.candidates_token_count, "total": usage.total_token_count}
        except Exception as e: 
            return UserIntent(categories=[], amount=0, reasoning=f"오류: {e}"), {"input":0, "output":0, "total":0}

    # [Page 1] 이력 기반 추천 
    def recommend_by_history(self, user_row: pd.Series, user_inventory: list = None):
        try:
            raw_dict = user_row.to_dict()
            numeric_spend = {}
            for k, v in raw_dict.items():
                if isinstance(v, (int, float, np.number)) and v > 0:
                    numeric_spend[k] = int(v)

            top_cats = sorted(numeric_spend.items(), key=lambda x: x[1], reverse=True)[:3]
            top_cat_names = ", ".join([k for k, v in top_cats])
            
            if self.retriever_minified: market_cards = self.retriever_minified.search(top_cat_names + " 혜택", top_k=30)
            else: market_cards = self.minified_db[:30]
                
            all_cards = []
            url_map = {}
            
            for idx, card in enumerate(user_inventory or []):
                cid = f"MY_{idx}"
                c_url, c_data = self._detach_url_from_card(card, cid)
                url_map[cid] = c_url
                all_cards.append({"type": "OWNED", "data": c_data})

            for idx, card in enumerate(market_cards):
                cid = f"NEW_{idx}"
                c_url, c_data = self._detach_url_from_card(card, cid)
                url_map[cid] = c_url
                all_cards.append({"type": "MARKET", "data": c_data})
                
            context_str = json.dumps(all_cards, ensure_ascii=False)
            
            column_legend = """
            - R3M_FOOD_AMT: 요식, R3M_ENT_AMT: 유흥, R3M_DEP_AMT: 백화점
            - R3M_MART_AMT: 대형마트, R3M_SSM_AMT: 슈퍼마켓, R3M_CONV_AMT: 편의점
            - R3M_CLOTHES_AMT: 의류, R3M_ACCO_AMT: 숙박, R3M_TRAVEL_AMT: 여행
            - R3M_TRANS_AMT: 교통, R3M_BEAUTY_AMT: 미용, R3M_HOUSEHOLD_AMT: 생활
            - R3M_EDU_AMT: 교육, R3M_MED_AMT: 의료, R3M_OIL_AMT: 주유
            - R3M_E_COMM_AMT: 전자상거래(온라인), R3M_DLV_AMT: 배달앱
            - R3M_STARBUCKS_AMT: 스타벅스, R3M_E_CHARGE_AMT: 전기차 충전
            """

            system_instr = f"""
            당신은 금융 데이터 분석가입니다. 소비 패턴을 분석하여 최적의 카드를 추천하세요.
            주어진 소비내역 데이터의 단위는 천원(1000)입니다.
            또한 해당 소비 데이터는 3개월 통합이므로 소비 내역/3으로 나눠서 월 혜택을 산출해야 합니다.
            [데이터 해석 가이드]
            - **t**: 카드명, **b**: 혜택, **v**: 값, **u**: 단위(percent/won), **l**: 한도
            [소비 데이터 컬럼 해석 (R3M = 최근 3개월)]
            {column_legend}
            
            [분석 규칙]
            1. '{top_cat_names}' 등 지출이 큰 카테고리의 혜택을 최우선으로 계산하세요.
            2. 상세한 혜택 계산 과정을 'benefit_formula' 필드에 반드시 작성하세요.
            3. 추천 순위는 혜택 금액이 높은순으로 정렬하세요.
            """
            
            prompt = f"[소비내역]: {json.dumps(numeric_spend, ensure_ascii=False)}\n[카드DB]: {context_str}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instr,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=CardAnalysisResponse
                )
            )
            return response.parsed, url_map
            
        except Exception as e:
            print(f"Error in recommend_by_history: {e}")
            return None, None


    # [Page 2] 쇼핑 혜택 계산 로직
    def calculate_benefits(self, intent: UserIntent, user_inventory: list = None):
        target_merchant = intent.target_merchant.lower() if intent.target_merchant else ""
        market_cards = self.batch1_db 

        def score_cards(cards, prefix="NEW"):
            scored = []
            user_wants_dept = any("department" in c or "백화점" in c for c in intent.categories)

            for idx, card in enumerate(cards):
                card_name = self.get_card_name(card).replace("네이버 신용카드 정보:", "").strip()
                cid = f"{prefix}_{idx}"
                card_url = self._generate_url(card)
                
                max_score = 0.0; final_benefit = 0.0
                best_desc = ""; best_logic = ""
                
                is_dept_card = "백화점" in card_name or "department" in card_name.lower()
                benefits = card.get("benefits") or card.get("b") or []
                
                for ben in benefits:
                    cat = ben.get("category") or ben.get("c") or ""
                    cat_list = cat if isinstance(cat, list) else [cat]
                    desc_text = (ben.get("desc") or ben.get("d") or "").lower()
                    val_raw = ben.get("value") or ben.get("v") or 0
                    limit_raw = ben.get("limit") or ben.get("l") or 0
                    unit = ben.get("unit") or ben.get("u") or "percent"

                    # 가중치 로직
                    relevance = 0
                    is_dept_benefit = "백화점" in desc_text or "department" in desc_text
                    if target_merchant and (target_merchant in desc_text or target_merchant in ["백화점", "department"]): relevance = 1.2
                    elif is_dept_card or is_dept_benefit: relevance = 0.5 
                    elif any(c in intent.categories for c in cat_list): relevance = 1.0
                    elif "all" in cat_list: relevance = 0.9

                    if relevance > 0:
                        try:
                            val = float(val_raw)
                            limit = float(99999999 if limit_raw == -1 else limit_raw)
                            if limit == 0: limit = 30000 
                        except: val, limit = 0, 0
                        
                        calc = intent.amount * (val / 100.0) if unit == "percent" else val
                        real_money = min(calc, limit)
                        ranking_score = real_money * relevance
                        
                        if ranking_score > max_score: 
                            max_score = ranking_score
                            final_benefit = real_money # [중요] 실제 금액은 가중치 없이 저장
                            best_desc = ben.get("desc") or ben.get("d") or ""
                            if unit == "percent": best_logic = f"결제금액 {intent.amount:,}원의 {val}% 할인 (한도 {limit:,}원)"
                            else: best_logic = f"고정 할인 {val:,}원 적용 (한도 {limit:,}원)"
                
                if max_score > 0:
                    scored.append({
                        "card_id": cid, "card_name": card_name, "benefit_amt": int(final_benefit),
                        "ranking_score": max_score, "benefit_desc": best_desc, "benefit_logic_raw": best_logic,
                        "card_url": card_url, "annual_fee": str(card.get('annual_fee') or card.get('af') or '정보없음'),
                        "_raw_card": card 
                    })
            return sorted(scored, key=lambda x: x['ranking_score'], reverse=True)

        my_results_all = score_cards(user_inventory, "MY") if user_inventory else []
        market_results_all = score_cards(market_cards, "NEW")
        
        # Reranking (Top 40 -> RRF -> Top 5)
        top_candidates = market_results_all[:40]
        search_query = f"{target_merchant} {' '.join(intent.categories)}".strip()

        if top_candidates and search_query:
            candidate_raw_data = [item['_raw_card'] for item in top_candidates]
            temp_retriever = HybridRetriever(candidate_raw_data, self.client)
            reranked_cards = temp_retriever.search(search_query, top_k=10)
            
            final_recs = []
            for r_card in reranked_cards:
                r_name = self.get_card_name(r_card).replace("네이버 신용카드 정보:", "").strip()
                for cand in top_candidates:
                    if cand['card_name'] == r_name: final_recs.append(cand); break
            
            if len(final_recs) < 5:
                existing_names = set(c['card_name'] for c in final_recs)
                remaining = [c for c in top_candidates if c['card_name'] not in existing_names]
                final_recs.extend(remaining[:5-len(final_recs)])
            market_final = final_recs[:5]
        else:
            market_final = top_candidates[:5]
        
        for item in my_results_all: item.pop('_raw_card', None); item.pop('ranking_score', None)
        for item in market_final: item.pop('_raw_card', None); item.pop('ranking_score', None)

        return {"my_cards": my_results_all[:3], "recs": market_final}

    # [Page 2] 리포트 생성 (정답지 활용)
    def generate_simulation_report(self, query: str, intent: UserIntent, results: dict):
        my_card = results["my_cards"]; recs = results["recs"]
        if not my_card and not recs: return None, None
        
        combined_list = my_card + recs
        url_map = {item['card_id']: item['card_url'] for item in combined_list}
        context_data = [{k: v for k, v in item.items() if k != 'card_url'} for item in combined_list]

        system_instr = """당신은 카드 혜택 시뮬레이션 리포트 작성자입니다.
        계산된 데이터를 JSON 스키마(CardAnalysisResponse)에 맞춰 변환하세요.
        [매핑 규칙]
        1. 입력 'card_name' -> 출력 'card_name'
        2. 입력 'benefit_amt' -> 출력 'expected_benefit'
        3. 입력 'annual_fee' -> 출력 'annual_fee'
        4. **입력된 'benefit_logic_raw' 정보를 그대로 활용하여 'benefit_formula' 필드에 계산 과정을 서술하세요.**
           (예: "결제금액 100,000원의 10% 할인이 적용되었습니다.")
        5. summary 필드: 추천 이유 3줄 요약.
        
        [검증 및 방어 로직]
        1. 질문의 의도와 추천된 카드의 혜택이 일치하는지 검증하세요.
        2. 만약 사용자가 특정 상점(예: 쿠팡)을 물어봤는데 결과가 '백화점 혹은 멤버십 전용' 카드라면, summary 필드에 **"요청하신 상점의 전용 혜택은 없으나, 기본 할인이 적용되어 추천되었습니다"**라고 명시하세요.
        """
        prompt = f"[질문]: {query}\n[계산결과]: {json.dumps(context_data, ensure_ascii=False)}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system_instr, temperature=0.1, response_mime_type="application/json", response_schema=CardAnalysisResponse)
            )
            return response.parsed, url_map
        except Exception: return None, None

    # [Page 3] 프리미엄 상담
    def recommend_by_prompt(self, user_input: str, user_inventory: list = None):
        try:
            intent, _ = self.analyze_intent(user_input)
            search_keywords = [user_input]
            if intent.categories: search_keywords.extend(intent.categories)
            search_query = " ".join(search_keywords) + " 혜택"
            
            # 1. 단순 검색 (Python 계산 로직 없음)
            if self.retriever_minified: market_cards = self.retriever_minified.search(search_query, top_k=20)
            else: market_cards = self.minified_db[:20]

            all_cards = []
            url_map = {}
            for idx, card in enumerate(user_inventory or []):
                cid = f"MY_{idx}"; c_url, c_data = self._detach_url_from_card(card, cid)
                url_map[cid] = c_url; all_cards.append({"type": "OWNED", "data": c_data})
            for idx, card in enumerate(market_cards):
                cid = f"NEW_{idx}"; c_url, c_data = self._detach_url_from_card(card, cid)
                url_map[cid] = c_url; all_cards.append({"type": "MARKET", "data": c_data})
                
            # 2. LLM에게 문맥 전체 전달 (Standard RAG)
            context_str = json.dumps(all_cards, ensure_ascii=False)
            system_instr = """
            당신은 대한민국 최고의 카드 혜택 컨설턴트입니다.

            [데이터 해석 가이드]
            - **t (Title)**: 카드 이름
            - **b (Benefits)**: 혜택 리스트
            - **v (Value)**: 혜택의 크기 (숫자). 
            - **u (Unit)**: 단위. 'percent'(%) 또는 'won'(원) 또는 'liter'(리터당 원).
            - **l (Limit)**: 월간 통합 할인 한도. (매우 중요: 아무리 %가 높아도 이 한도를 넘을 수 없음)
            당신은 프리미엄 카드 컨설턴트입니다. 사용자 질문을 분석하여 최적의 카드를 추천하고 JSON으로 응답하세요.

            [분석 규칙]
            1. 'card_name'에는 실제 카드 이름을 사용하세요. (시스템 ID 사용 금지)
            2. 질문에 포함된 특정 상점이나 카테고리(예: 스타벅스, 주유소)의 혜택을 정밀 계산하세요.
            3. 'pe'(실적제외) 조건이 까다로운 카드는 추천 순위를 낮추세요.
            4. 상세한 혜택 계산 공식을 'benefit_formula' 필드에 작성하세요.
            5. 지정된 JSON 스키마(CardAnalysisResponse)에 맞춰 데이터만 반환하세요.

            [지침]
            1. 사용자의 텍스트에서 소비 금액을 추출하고, 카드의 혜택 공식(v, l, u)에 대입하여 '월 예상 절약 금액'을 산출하세요.
            2. **"월 예상 혜택: OOO원"** 형식을 반드시 지키세요.
            3. 보유 카드(OWNED_CARD)와 시장 카드(MARKET_CARD)의 혜택 금액을 비교하세요.
            5. 정확한 계산식을 반드시 제시하세요.
            6. 최적의 카드 조합을 추천하세요.
            """
            prompt = f"[사용자요청]: {user_input}\n[카드DB]: {context_str}"
            
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system_instr, temperature=0.1, response_mime_type="application/json", response_schema=CardAnalysisResponse)
            )
            return response.parsed, url_map
        except Exception: return None, None