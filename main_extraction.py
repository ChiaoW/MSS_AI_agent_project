import json
import os
import re
from typing import List, Dict
from datetime import datetime

# LangChain Core components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI # 用來連線 vllm (OpenAI Compatible)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
import json_repair

from file_processor import UniversalFileProcessor
from pydantic_schema import OrderInfo, SampleInfo
from rag_retriever import DynamicFewShotRetriever

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "semiconductor_orders_english"
LLM_API_BASE = "http://localhost:8000/v1"
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MODEL_NAME = "openai/gpt-oss-20b"

def truncate_text(text: str, max_chars: int = 60000) -> str:
    if len(text) <= max_chars:
        return text
    print('Text Truncated')
    return text[:max_chars] + "\n...[Content Truncated due to length limit]..."

def clean_and_extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        return text[start_idx : end_idx + 1]
    
    # Fallback if no brackets are found: wrap it in empty JSON so the pipeline doesn't crash
    return '{"global_analysis": "Failed to generate JSON", "samples": []}'

def process_lot_request(lot_directory: str, lot_id_override: str = None):

    processor = UniversalFileProcessor()
    print(f"Processing files in {lot_directory}...")
    try:
        full_context_text = processor.process_directory(lot_directory)
    except Exception as e:
        print(f"Error reading files: {e}")
        return None


    input_text = truncate_text(full_context_text, max_chars=40000)

    # Determine lot_id for debug file naming
    if lot_id_override is not None:
        debug_lot_id = lot_id_override
    else:
        debug_lot_id = os.path.basename(lot_directory)

    # def build_smart_query(text: str) -> str:
    #     """
    #     過濾掉問候語、免責聲明等雜訊，只提取包含半導體術語的行，
    #     組成高密度的查詢字串，以提升 Embedding 檢索的精準度。
    #     """
    #     # 定義高價值的領域關鍵字 (全小寫)
    #     keywords = [
    #         "sample", "wafer", "route", "prepare", "cut", "tem", "fib", 
    #         "ald", "db", "epoxy", "probing", "condition", "recipe", 
    #         "x-cut", "y-cut", "thickness", "target", "lot", "die", "layer", "pad", "eds",
    #         "process", "flow", "analysis", "test", "req", "require"
    #     ]
        
    #     lines = text.split('\n')
    #     important_lines = []
        
    #     for line in lines:
    #         # 只要該行包含任何一個關鍵字，就保留
    #         if any(k in line.lower() for k in keywords):
    #             important_lines.append(line.strip())
        
    #     # 將這些高價值行組合起來
    #     condensed_text = "\n".join(important_lines)
        
    #     # 限制在 8000 字元內 (確保符合 Jina 8192 token 模型上限)
    #     return condensed_text[:8000]

    # 產生專門用來檢索的濃縮查詢
    # search_query = build_smart_query(input_text)

    search_query = input_text[:8000]

    try:
        retriever = DynamicFewShotRetriever(DB_PATH)
        few_shot_examples = retriever.get_few_shot_examples(search_query, k=4)
        print(f"Retrieved {len(few_shot_examples)} similar past cases.")

        print("\n[RAG Debug] Selected Few-shot Examples (Full Answers):")
        for idx, ex in enumerate(few_shot_examples):
            print(f"  Example #{idx + 1}")
            
            # 印出完整答案
            full_answer = ex.get('answer', '')
            print(f"    Answer Content:\n{full_answer}")
            print("-" * 30)
        print("==========================================\n")

    except Exception as e:
        print(f"RAG Error (fallback to empty examples): {e}")
        few_shot_examples = []

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "=== HISTORICAL REFERENCE CASE (DO NOT EXTRACT WAFER IDs FROM THIS) ===\n\n{context_text}\n\n=== END OF HISTORICAL CASE ==="),
            ("ai", "{answer}"),
        ]
    )

    # 建立 Few-Shot Template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shot_examples, # 這裡是動態注入的 RAG 結果
    )
    
    system_instruction = (
        "You are an expert semiconductor order extraction agent. "
        "Your task is to extract information into the requested JSON schema.\n\n"

        "### CORE RULES:\n"
        "1. **VENDOR ROLE:** You represent MSS. The 'customer' is the external client (e.g., TEL). NEVER list MSS, Panquan, or internal employees (Katie, Chen Jiaxin, David, Amy) as the customer_name.\n"
        "2. **CURRENT CASE ONLY:** Extract data ONLY from the '=== CURRENT TARGET CASE ===' section. Do NOT extract Wafer IDs from the historical examples.\n"
        "3. **SAMPLE MATCHING:** Use 'Navi map' or 'Macro' names to match the exact Wafer IDs in the table to the engineer's instructions in the emails.\n\n"

        "### HOW TO BUILD THE 'PREPARE' FIELD (Dynamic SOP Logic):\n"
        "- **Explicit Parameters:** If the current case PPT/email provides exact parameters (e.g., 'ALD(W2-A)', 'Pi-bond(60/30)'), use them exactly.\n"
        "- **Vague Parameters (History Fallback):** If the email uses vague terms like 'ALD Coating', 'HfO2', 'Epoxy', or 'Glue', you MUST look at how the 'HISTORICAL REFERENCE CASES' handled similar macros. \n"
        "  - E.g., If the customer asks for 'Epoxy', look at the historical cases. If they previously used 'Pi bond(60/30)' for this macro, output 'Pi bond(60/30)'. If they used 'M-bond(60/30)', output 'M-bond(60/30)'.\n"
        "  - E.g., If they ask for 'ALD', look at the history to find the exact cycle count (like 'W2 35cycle').\n"
        "- **Additional Steps:**\n"
        "  - If 'Topview' or 'Planview' is requested, add '+ Top view'.\n"
        "  - If 'DB-located' or 'DB positioning' is requested, add '+ DB positioning'.\n"
        "  - If 'Probing' is requested, add '+ Probing'.\n\n"

        "### HOW TO BUILD THE 'ROUTE' FIELD:\n"
        "- If 'Probing' is required for the sample -> Output exactly 'ALD+Probing'.\n"
        "- If 'Topview' is required (but no probing) -> Output exactly 'ALD+TOP view+ TEM'.\n"
        "- If neither is required -> Output exactly 'ALD+normal' or 'FIB_XS ALD+ Normal FIB'.\n"
        "- Do NOT invent routes. Output exactly ONE valid string.\n"
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            # Format instructions are injected safely here, away from the document body
            ("system", system_instruction + "\n\n### OUTPUT FORMAT INSTRUCTIONS ###\n{format_instructions}"),
            ("human", "Here are some reference examples from the database. Pay attention to how they map instructions to samples, but DO NOT EXTRACT their Wafer IDs:"),
            few_shot_prompt, 
            ("human", "=== CURRENT TARGET CASE STARTS HERE (EXTRACT THIS ONLY) ===\n\n{input}\n\n=== CURRENT TARGET CASE ENDS HERE ==="),
        ]
    )

    # 4. LLM & Parser Setup
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=LLM_API_BASE,
        openai_api_key="EMPTY",
        temperature=0, # 資訊萃取務必設為 0
        top_p=0.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=8192
    )
    
    parser = PydanticOutputParser(pydantic_object=OrderInfo)
    
    # 注入 Format Instructions (告訴 LLM JSON Schema)
    format_instructions = parser.get_format_instructions()
    
    # 建立 Chain
    chain = final_prompt | llm

    # [Debug Step 1] 組合最終 Prompt 內容並存檔檢查
    final_input_payload = {
        "input": input_text
    }
    
    # 這裡我們手動 format 一次來看看 (因為 chain 封裝了 prompt，看不到組合後的樣子)
    debug_prompt_str = final_prompt.format(
        input=final_input_payload["input"],
        format_instructions=format_instructions
    )
    
    with open(f"data/output/three_cases_debug_prompt/DEBUG_FINAL_PROMPT_{debug_lot_id}.txt", "w", encoding="utf-8") as f:
        f.write(debug_prompt_str)
        
    print(f"[Debug] Final prompt saved to DEBUG_FINAL_PROMPT.txt. Please check it!")

    print("Invoking LLM for extraction...")

    try:
        # Pass input_text and format_instructions separately to the prompt template
        # The prompt template will handle combining them properly
        
        raw_response = chain.invoke({
            "input": input_text,
            "format_instructions": format_instructions
        })
    
        if hasattr(raw_response, 'content'):
            raw_content = raw_response.content
        else:
            raw_content = str(raw_response)

        # [Debug] 印出前 500 字確認內容
        print(f"\n[Raw LLM Output Preview]:\n{raw_content[:500]}...\n")

        raw_content = clean_and_extract_json(raw_content)

        print("Parsing JSON output...")
        try:
            data = json_repair.loads(raw_content)
            
            # Bulletproof fallback if it's still a string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    print(f"  [JSON Error] Could not parse string into dict. Forcing empty dict.")
                    data = {"samples": []}
            
            if not isinstance(data, dict):
                print(f"  [Critical] Parsed output is type {type(data)}. Forcing empty dict.")
                data = {"samples": []}

        except Exception as e:
            print(f"  [JSON Error] JSON parsing failed: {e}")
            data = {"samples": []}

        # B. 逐筆驗證 Samples (過濾掉爛尾的項目)
        valid_samples = []
        raw_samples = data.get("samples", [])
        
        if isinstance(raw_samples, list):
            for i, item in enumerate(raw_samples):
                try:
                    # 嘗試將這個 dict 轉為 SampleInfo 物件
                    # 這會觸發我們寫的所有 Validators (Route check, Epoxy fix 等)
                    sample_obj = SampleInfo(**item)
                    valid_samples.append(sample_obj)
                except ValidationError as ve:
                    # 這裡就是攔截 "Field required" 的地方
                    print(f"  [Skipping] Invalid sample #{i+1}: Missing fields or validation failed.")
                    print(f"    Reason: {ve}") # 需要詳細除錯再打開
                    continue
        
        # C. 重新組裝 OrderInfo
        # 即使 data 裡面的 global_analysis 也是空的，給個預設值，保證流程能往下走
        try:
            final_order = OrderInfo(
                global_analysis=data.get("global_analysis", "Analysis text missing or truncated"),
                company=data.get("company"),
                customer_name=data.get("customer_name"),
                samples=valid_samples # 這裡只放成功的 samples
            )
            
            print(f"\n=== Extraction Result (Recovered {len(valid_samples)}/{len(raw_samples)} samples) ===")
            
            # [Post-processing] Lot ID 重新編號 (因為可能有 sample 被跳過，重新排序比較保險)
            lot_base_name = os.path.basename(lot_directory) 
            for idx, sample in enumerate(final_order.samples):
                suffix = f"{idx + 1:03d}"
                sample.lot_id = f"{lot_base_name}-{suffix}"

            print(final_order.model_dump_json(indent=2))
            return final_order

        except ValidationError as ve:
            print(f"  [Critical] OrderInfo Level Validation Failed: {ve}")
            return None
    
    except Exception as e:
        print(f"LLM Extraction Failed: {e}")
        return None

if __name__ == "__main__":
    # Batch test: run three cases and save outputs to a timestamped file
    cases = [
        "data/raw/all_cases/T25100101",
        "data/raw/Kang_Yi_Lin_Merged/T25082545",
        "data/raw/Kang_Yi_Lin_Merged/T251020101",
    ]

    results = []

    # Load ground-truth answer files (if present)
    gt_all_cases = []
    gt_kyl = {}
    all_cases_gt_path = os.path.join("answers", "answer_with_wafer_id_processed.json")
    kyl_gt_path = os.path.join("answers", "Kang_Yi_Lin_case_ground_truth_translated.json")

    if os.path.exists(all_cases_gt_path):
        try:
            with open(all_cases_gt_path, "r", encoding="utf-8") as f:
                gt_all_cases = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {all_cases_gt_path}: {e}")

    if os.path.exists(kyl_gt_path):
        try:
            with open(kyl_gt_path, "r", encoding="utf-8") as f:
                gt_kyl = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {kyl_gt_path}: {e}")

    def find_ground_truth_for_lot(lot_id: str):
        """Return list of ground-truth sample dicts that match the given full lot_id."""
        matches = []
        # 1) Search in gt_all_cases (list of entries with 'samples')
        if isinstance(gt_all_cases, list):
            for entry in gt_all_cases:
                for s in entry.get("samples", []) or []:
                    if s.get("lot_id") == lot_id:
                        # include some context (mail_id/email_subject) if available
                        record = {**s}
                        if entry.get("mail_id") is not None:
                            record["_mail_id"] = entry.get("mail_id")
                        if entry.get("email_subject") is not None:
                            record["_email_subject"] = entry.get("email_subject")
                        matches.append(record)

        # 2) Search in gt_kyl (mapping from base lot to samples list)
        if isinstance(gt_kyl, dict):
            # keys may be base lot ids like 'T25082545'
            for base_lot, obj in gt_kyl.items():
                # If the predicted lot_id begins with the base key, check its samples
                if lot_id.startswith(base_lot):
                    for s in obj.get("samples", []) or []:
                        if s.get("lot_id") == lot_id:
                            matches.append(s)

        return matches

    for case_dir in cases:
        if os.path.exists(case_dir):
            print(f"\n=== Processing case: {case_dir} ===")
            try:
                # Use the last part of the directory as lot_id for debug file naming
                lot_id_for_debug = os.path.basename(case_dir)
                res = process_lot_request(case_dir, lot_id_override=lot_id_for_debug)
                if res is None:
                    results.append((case_dir, "<No result or parsing failed>"))
                else:
                    # Use Pydantic's JSON output for structured result
                    try:
                        pred_json = res.model_dump_json(indent=2)
                    except Exception:
                        pred_json = str(res)

                    # Attempt to parse prediction JSON to extract predicted lot_ids
                    gt_matches = {}
                    try:
                        pred_obj = json.loads(pred_json)
                        for s in pred_obj.get("samples", []):
                            lot = s.get("lot_id")
                            if lot:
                                gt = find_ground_truth_for_lot(lot)
                                if gt:
                                    gt_matches[lot] = gt
                    except Exception:
                        pred_obj = None

                    # Build combined content: prediction then ground-truth matches
                    combined = {
                        "prediction": json.loads(pred_json) if pred_obj is not None else pred_json,
                        "ground_truth_matches": gt_matches
                    }
                    results.append((case_dir, json.dumps(combined, indent=2, ensure_ascii=False)))
            except Exception as e:
                results.append((case_dir, f"<Processing error: {e}>"))
        else:
            results.append((case_dir, "<Case folder not found>"))

    # Save aggregated results to a timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"data/output/three_cases_predict_results/results_{ts}.txt"
    with open(out_fname, "w", encoding="utf-8") as outf:
        for case_dir, content in results:
            outf.write(f"=== CASE: {case_dir} ===\n")
            outf.write(content)
            outf.write("\n\n")

    print(f"Batch run complete. Results saved to {out_fname}")