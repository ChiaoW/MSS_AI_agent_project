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

    # Save full context text for debugging
    os.makedirs("data/output/debug_prompt", exist_ok=True)
    with open(f"data/output/debug_prompt/{debug_lot_id}.txt", "w", encoding="utf-8") as f:
        f.write(full_context_text)

    # llm = ChatOpenAI(
    #     model=MODEL_NAME,
    #     openai_api_base=LLM_API_BASE,
    #     openai_api_key="EMPTY",
    #     temperature=0, 
    #     top_p=0.1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     max_tokens=8192
    # )

    # # 新增：利用 LLM 產生高密度的 Retrieval Query
    # print("Generating dense retrieval query via LLM...")
    # query_prompt = ChatPromptTemplate.from_template(
    #     "You are an expert semiconductor engineer. Extract the core technical requirements "
    #     "from the following text (e.g., required processes like TEM, ALD, FIB, probing, "
    #     "DB positioning, thickness, and specific materials like Epoxy or Pi-bond). "
    #     "Output ONLY a concise, comma-separated list of technical keywords and requirements. "
    #     "Do NOT include greetings, email boilerplate, or conversational text.\n\n"
    #     "Text:\n{text}"
    # )
    
    # query_chain = query_prompt | llm
    
    # try:
    #     # 只取前 8000 字元給 LLM 總結，避免超過 Context Window，也節省運算
    #     summary_response = query_chain.invoke({"text": input_text[:8000]})
    #     search_query = summary_response.content.strip()
    #     print(f"  [Smart Query Generated]: {search_query}")
    # except Exception as e:
    #     print(f"  [Smart Query Error]: {e}. Falling back to raw text.")
    #     search_query = input_text[:8000]

    search_query = input_text[:8000]

    try:
        retriever = DynamicFewShotRetriever(DB_PATH)
        few_shot_examples = retriever.get_few_shot_examples(search_query, k=3)
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
        "3. For each sample, write your step-by-step logic in the thought_process field before determining the route and prepare values.\n"
        "4. **SAMPLE MATCHING:** Use 'Navi map' or 'Macro' names to match the exact Wafer IDs in the table to the engineer's instructions in the emails.\n\n"

        "### HOW TO BUILD THE 'PREPARE' FIELD (Dynamic SOP Logic):\n"
        "- **Explicit Parameters:** If the current case PPT/email provides exact parameters (e.g., 'ALD([Specific Cycle])', '[Specific Material]-bond([Ratio])'), use them exactly as written in the target text.\n"
        "- **Vague Parameters (History Fallback):** If the email uses vague terms like 'ALD Coating', 'HfO2', 'epoxy', or 'Glue', you MUST look at how the 'HISTORICAL REFERENCE CASES' handled similar macros. \n"
        "  - E.g., If the customer asks for 'epoxy', **MUST** look at the historical cases. If they previously used 'Pi bond(60/30)' for this macro, output 'Pi bond(60/30)'. If they used 'M-bond(60/30)', output 'M-bond(60/30)'.\n"
        "  - E.g., If they ask for 'ALD', look at the history to find the exact cycle count (like 'W2 35cycle').\n"
        "- **Additional Steps:**\n"
        "  - If 'Topview' or 'Planview' is requested, add '+ Top view'.\n"
        "  - If 'DB-located' or 'DB positioning' is requested, add '+ DB'.\n"
        "  - If 'Probing' is requested, add '+ Probing'.\n\n"

        "### HOW TO BUILD THE 'ROUTE' FIELD:\n"
        "- Do NOT invent routes. Output exactly ONE valid string.\n"
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "Here are some reference examples from the database. Pay attention to how they map instructions to samples, but DO NOT EXTRACT their Wafer IDs:\n<historical_examples>"),
            few_shot_prompt, 
            ("human", "</historical_examples>\n\n<target_case>\n{input}\n</target_case>\n\nCRITICAL: Extract data strictly from within the <target_case> tags. Do not use parameters from the historical examples."),
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
    # 啟用 API 層級的 Structured Outputs，將 OrderInfo Schema 綁定到模型
    structured_llm = llm.with_structured_output(OrderInfo)
    
    # 建立 Chain (不需要 parser)
    chain = final_prompt | structured_llm

    # [Debug Step 1] 組合最終 Prompt 內容並存檔檢查
    debug_prompt_str = final_prompt.format(input=input_text)
    
    with open(f"data/output/three_cases_debug_prompt/DEBUG_FINAL_PROMPT_{debug_lot_id}.txt", "w", encoding="utf-8") as f:
        f.write(debug_prompt_str)
        
        f.write("\n\n" + "="*50 + "\n")
        f.write("[DEBUG] FULL FEW-SHOT ANSWERS\n")
        f.write("="*50 + "\n")
        for idx, ex in enumerate(few_shot_examples):
            f.write(f"\n--- Example #{idx + 1} Answer ---\n")
            f.write(str(ex.get('answer', '')))
        
    print(f"[Debug] Final prompt saved to DEBUG_FINAL_PROMPT_{debug_lot_id}..txt. Please check it!")

    print("Invoking LLM with Structured Outputs for precise extraction...")

    try:
        # 直接執行 invoke，回傳的就會是經過嚴格驗證的 OrderInfo Pydantic 物件
        final_order = chain.invoke({"input": input_text})

        if not final_order:
            print("  [Critical] Model failed to return valid structured output.")
            return None
        
        # 過濾掉可能因為幻覺產生但內容完全空洞的 sample
        valid_samples = [s for s in final_order.samples if s.wafer_id]

        print(f"\n=== Extraction Result (Recovered {len(valid_samples)} samples) ===")
        
        # [Post-processing] Lot ID 重新編號
        lot_base_name = os.path.basename(lot_directory) 
        for idx, sample in enumerate(valid_samples):
            suffix = f"{idx + 1:03d}"
            sample.lot_id = f"{lot_base_name}-{suffix}"

        # 寫回過濾後的 samples
        final_order.samples = valid_samples

        print(final_order.model_dump_json(indent=2))
        return final_order
    
    except Exception as e:
        print(f"LLM Extraction with Structured Outputs Failed: {e}")
        return None

if __name__ == "__main__":
    # Batch test: run three cases and save outputs to a timestamped file
    cases = [
        "data/raw/all_cases/T25100101",
        # "data/raw/Kang_Yi_Lin_Merged/T25082545",
        # "data/raw/Kang_Yi_Lin_Merged/T251020101",
    ]

    results = []

    # Load ground-truth answer files (if present)
    gt_all_cases = []
    gt_kyl = {}
    all_cases_gt_path = os.path.join("data/reference/answers", "answer_with_wafer_id_processed.json")
    kyl_gt_path = os.path.join("data/reference/answers", "Kang_Yi_Lin_case_ground_truth_translated.json")

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