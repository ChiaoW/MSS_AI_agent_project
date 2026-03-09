import os
import json
import re
from datetime import datetime
import logging
import dspy

from src.rag_retriever import DynamicFewShotRetriever
from src.pydantic_schema import OrderInfo, SampleInfo, Stage1Order, Stage2Inference, Stage1Sample, active_routes

logger = logging.getLogger(__name__)

class CustomMssRM(dspy.Retrieve):
    def __init__(self, db_url: str, k: int = 3):
        super().__init__(k=k)
        # 實例化你原本寫好的 Retriever
        self.internal_retriever = DynamicFewShotRetriever(db_url)
    
    def forward(self, query_or_queries, k=None, **kwargs):
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        passages = []
        for q in queries:
            # 呼叫你原本在 rag_retriever.py 寫好的 get_few_shot_examples
            examples = self.internal_retriever.get_few_shot_examples(q, k=k)
            for ex in examples:
                # 將 RAG 抓出來的結果，組合成純文字讓 DSPy 語言模型看懂
                doc_text = f"Context: {ex['context_text']}\nOutput JSON: {ex['answer']}"
                passages.append(doc_text)
                
        # DSPy 的底層要求回傳帶有 'long_text' 屬性的字典清單
        return [dspy.Prediction(long_text=p) for p in passages]
    
class Stage1Signature(dspy.Signature):
    """
    You are an expert semiconductor entity extraction agent.
    Your ONLY task is to extract the customer metadata and list ALL unique Wafer IDs / Sample IDs present.
    
    RULES:
    1. VENDOR ROLE: You represent MSS. The 'customer' and 'company' refer to external clients. NEVER list MSS or internal employees.
    2. LEARNING FROM HISTORY: Look at the provided historical examples to understand what a valid 'Wafer ID' looks like. DO NOT extract Wafer IDs from the historical cases.
    3. MULTIPLE TESTS: If the same Wafer ID has multiple different test items, locations, or routes, you MUST extract it multiple times as separate entries. Do not omit duplicates if they represent distinct physical tests.
    4. FALLBACK ID: If there is no standard Wafer ID code, but there is a generic description (e.g., "Wafer 1"), extract that as the Wafer ID so the process can continue.
    5. BRIEF REASONING: Keep your reasoning strictly under 50 words. Do not over-explain.

    EXPECTED JSON FORMAT:
    {
        "thought_process": "Your step-by-step reasoning logic for finding the entities...",
        "company": "extracted company name",
        "customer": "extracted customer name",
        "wafer_ids": ["Wafer1", "Wafer2"]
    }
    """
    input_text: str = dspy.InputField(desc="The complete text of the original case (Target Case)")
    historical_examples: str = dspy.InputField(desc="Historical reference cases from the database (please refer to their format, but do not extract the IDs).")
    output_schema_instructions: str = dspy.InputField(desc="CRITICAL: You MUST structure your JSON output exactly according to this JSON Schema and obey the descriptions inside.")
    output: str = dspy.OutputField(desc="CRITICAL: Output ONLY valid JSON matching the schema above. Do not use markdown blocks (```json).")
    # output: Stage1Order = dspy.OutputField(desc="The JSON object containing reasoning, company, customer, and wafer IDs")

class Stage2Signature(dspy.Signature):
    """
    You are an expert semiconductor order logic inference agent.
    Your task is to analyze the ENTIRE provided case text, but extract the exact technical parameters ONLY for the specific TARGET WAFER ID requested.
    
    RULES:
    - CRITICAL: Output ONLY valid JSON. Do not use markdown blocks like ```json. Do not explain.
    - Explicit Parameters: Use exact parameters if present (e.g., 'ALD(W2-A)', 'M-bond(60/30)').
    - Vague Parameters (History Fallback): If the text uses vague terms (like 'epoxy' or 'ALD'), you MUST look at the historical examples to find the specific recipe used for similar macros.
    - Extra steps: Add '+ Top view', '+ DB', or '+ Probing' if explicitly requested.
    - Route: Select the exact matching standard process route code.

    EXPECTED OUTPUT FORMAT:
    {
        "thought_process": "Your step-by-step reasoning logic...",
        "route": "extracted route string",
        "prepare": ["step1", "step2"],
        "loctestkey": "location string"
    }
    """
    full_context: str = dspy.InputField(desc="The complete text of the original case (Target Case)")
    target_wafer_id: str = dspy.InputField(desc="The specific Wafer ID for which you need to infer the parameters")
    historical_examples: str = dspy.InputField(desc="Historical reference cases (used to infer the true setting of fuzzy parameters)")
    valid_routes: str = dspy.InputField(desc="List of allowed active routes. You MUST choose exactly one route from this list.")
    output_schema_instructions: str = dspy.InputField(desc="CRITICAL: You MUST structure your JSON output exactly according to this JSON Schema and obey the descriptions inside. Keep your thought process strictly under 50 words.")
    output: str = dspy.OutputField(desc="CRITICAL: Output ONLY valid JSON matching the schema above. Do not use markdown blocks (```json).")
    # output: Stage2Inference = dspy.OutputField(desc="The JSON object containing reasoning, route, prepare, and loctestkey")

class SemiconductorExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 ChainOfThought 來強制模型輸出 Pydantic 結構，並加上思考過程
        self.stage1 = dspy.Predict(Stage1Signature)
        self.stage2 = dspy.Predict(Stage2Signature)
        # self.retriever = dspy.Retrieve(k=3)

        self.latest_input = ""
        self.latest_context = ""

    def forward(self, input_text: str, lot_base_name: str = "UnknownLot"):
        self.latest_input = input_text
        self.latest_context = "尚未檢索 (Retrieval Failed)"

        # --- 步驟 1: RAG 檢索 ---
        retriever = dspy.Retrieve(k=3)
        search_query = input_text
        search_results = retriever(search_query)
        
        historical_context = ""
        for idx, passage in enumerate(search_results.passages):
            truncated_passage = str(passage)
            historical_context += f"=== HISTORICAL CASE #{idx+1} ===\n{truncated_passage}\n\n"

        self.latest_context = historical_context

        stage1_schema_str = json.dumps(Stage1Order.model_json_schema(), indent=2, ensure_ascii=False)

        # --- 步驟 2: 階段一 (實體切分) ---
        logger.info("\n[DSPy] Stage 1: Entity Extraction...")
        s1_pred = self.stage1(
            input_text=input_text, 
            historical_examples=historical_context,
            output_schema_instructions=stage1_schema_str
        )

        s1_raw_output = getattr(s1_pred, "output", "")
        
        # 進行手動清理與解析 (如同先前的 Retry 邏輯)
        cleaned_s1 = re.sub(r"<tool_call>.*?<tool_call>", "", str(s1_raw_output), flags=re.DOTALL | re.IGNORECASE)
        cleaned_s1 = re.sub(r"```json\s*", "", cleaned_s1, flags=re.IGNORECASE)
        cleaned_s1 = re.sub(r"```", "", cleaned_s1)
        
        start_idx = cleaned_s1.find('{')
        end_idx = cleaned_s1.rfind('}')
        
        stage1_samples = []
        global_analysis, company, customer_name = "", "", ""

        if start_idx != -1 and end_idx != -1:
            try:
                parsed_dict = json.loads(cleaned_s1[start_idx : end_idx + 1])
                stage1_samples = [Stage1Sample(**s) for s in parsed_dict.get("samples", [])]
                global_analysis = parsed_dict.get("global_analysis", "")
                company = parsed_dict.get("company", "")
                customer_name = parsed_dict.get("customer_name", "")
                logger.info("  [Success] 手動 JSON 清理與解析成功！")
            except Exception as e:
                logger.error(f"  [Critical] Stage 1 JSON 解析失敗: {e}")
                logger.info(f"  原始輸出: {s1_raw_output}")
                return None
        else:
            logger.error(f"  [Critical] Stage 1 找不到有效 JSON。原始輸出: {s1_raw_output}")
            logger.info(f"  s1_pred: {s1_pred}")
            logger.info(f"  原始輸出: {s1_raw_output}")
            return None

        if not stage1_samples:
             logger.error("  [Critical] 擷取到的 samples 列表為空。")
             logger.info(f"  原始輸出: {s1_raw_output}")
             return None

        # --- 步驟 3: 階段二 (單一樣本推論) ---
        logger.info("\n[DSPy] Stage 2: Logic Inference...")
        final_samples = []

        # 動態取得 Pydantic Schema 並轉為字串
        stage2_schema_str = json.dumps(Stage2Inference.model_json_schema(), indent=2, ensure_ascii=False)
        allowed_routes_str = ", ".join(active_routes)
        
        for idx, raw_sample in enumerate(stage1_samples):
            if not raw_sample.wafer_id:
                continue
                
            logger.info(f"  -> Inferring parameters for Wafer ID: {raw_sample.wafer_id}")
            suffix = f"{idx + 1:03d}"

            max_retries = 3
            route, prepare, loctestkey = None, None, None
            extracted_reasoning = ""
            
            for attempt in range(max_retries):
                try:
                    s2_pred = self.stage2(
                        full_context=input_text,
                        target_wafer_id=raw_sample.wafer_id,
                        historical_examples=historical_context,
                        valid_routes=allowed_routes_str,
                        output_schema_instructions=stage2_schema_str # 注入 Schema
                    )
                    
                    raw_text2 = getattr(s2_pred, "output", "")
                    
                    if not raw_text2:
                        logger.warning(f"  [Attempt {attempt+1}] Stage 2 output is empty. Retrying...")
                        continue

                    cleaned2 = re.sub(r"<think>.*?</think>", "", str(raw_text2), flags=re.DOTALL | re.IGNORECASE)
                    cleaned2 = re.sub(r"```json\s*", "", cleaned2, flags=re.IGNORECASE)
                    cleaned2 = re.sub(r"```", "", cleaned2)
                    
                    start_idx = cleaned2.find('{')
                    end_idx = cleaned2.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        parsed_dict = json.loads(cleaned2[start_idx : end_idx + 1])
                        validated_inference = Stage2Inference(**parsed_dict)
                        
                        route = validated_inference.route
                        prepare = validated_inference.prepare
                        loctestkey = validated_inference.loctestkey
                        extracted_reasoning = parsed_dict.get("thought_process", "")
                        
                        logger.info(f"  [Success] Wafer {raw_sample.wafer_id} 解析成功！")
                        break
                        
                except Exception as e:
                    logger.warning(f"  [Attempt {attempt+1}] 解析或驗證失敗: {e}")
                    logger.info(f"  原始輸出: {raw_text2}")
            
            if route is None and prepare is None:
                logger.error(f"  [Critical] Wafer {raw_sample.wafer_id} 推論失敗。")
                logger.info(f"  原始輸出: {raw_text2}")
                route, prepare, loctestkey = None, None, None

            final_sample = SampleInfo(
                lot_id=f"{lot_base_name}-{suffix}",
                wafer_id=raw_sample.wafer_id,
                thought_process=extracted_reasoning,
                route=route,
                prepare=prepare,
                loctestkey=loctestkey
            )
            final_samples.append(final_sample)

        final_order = OrderInfo(
            global_analysis=global_analysis,
            company=company,
            customer_name=customer_name,
            samples=final_samples
        )
        
        return dspy.Prediction(
            final_order=final_order, 
            historical_context=historical_context
        )
    
# Haven't used this yet.
def save_debug_prompt(lot_id: str, llm_instance, input_text: str, historical_context: str):
    """將 Input, Context 與 LLM 互動紀錄完整存入檔案"""
    output_dir = "data/output/debug_prompt"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/dspy_{lot_id}_{ts}.txt"
    
    history = llm_instance.history
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== Debug Log for Lot: {lot_id} ===\n")
        f.write(f"Total LLM Calls: {len(history)}\n\n")
        
        # 1. 寫入原始處理後的文字 (多檔案合併結果)
        f.write("="*20 + " 1. INPUT TEXT (File Contents) " + "="*20 + "\n")
        f.write(str(input_text) + "\n\n")

        # 2. 寫入 RAG 抓取到的案例
        f.write("="*20 + " 2. SELECTED FEW-SHOT EXAMPLES " + "="*20 + "\n")
        f.write(str(historical_context) + "\n\n")

        # 3. 寫入模型對話紀錄
        f.write("="*20 + " 3. ACTUAL LLM INTERACTIONS " + "="*20 + "\n")
        for i, interaction in enumerate(history):
            f.write(f"\n{'='*15} LLM CALL #{i+1} {'='*15}\n")
            f.write(">>> EXACT PROMPT SENT TO LLM:\n")
        
            prompt = interaction.get('prompt')
            if prompt is None:
                kwargs = interaction.get('kwargs', {})
                messages = kwargs.get('messages')
                if messages:
                    prompt = json.dumps(messages, indent=2, ensure_ascii=False)
                else:
                    prompt = "N/A (Prompt data not found in history)"
                    
            f.write(str(prompt) + "\n")
            
            f.write("\n<<< LLM RAW RESPONSE:\n")
            response_obj = interaction.get('response', '')
            if isinstance(response_obj, list) and len(response_obj) > 0:
                if hasattr(response_obj[0], 'message'):
                    f.write(str(response_obj[0].message.content) + "\n")
                else:
                    f.write(str(response_obj) + "\n")
            else:
                f.write(str(response_obj) + "\n")
            f.write("\n\n")
            
    logger.info(f"  [Debug] 實際對話紀錄已儲存至: {filename}")
    llm_instance.history.clear()