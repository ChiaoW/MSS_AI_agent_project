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
from pydantic_schema import OrderInfo, SampleInfo, Stage1Order, Stage2Inference
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
        return None, None

    input_text = truncate_text(full_context_text, max_chars=40000)

    if lot_id_override is not None:
        debug_lot_id = lot_id_override
    else:
        debug_lot_id = os.path.basename(lot_directory)

    # Save full context text for debugging
    os.makedirs("data/output/debug_prompt", exist_ok=True)
    with open(f"data/output/debug_prompt/{debug_lot_id}.txt", "w", encoding="utf-8") as f:
        f.write(full_context_text)

    # 初始化統一的 LLM 實例
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=LLM_API_BASE,
        openai_api_key="EMPTY",
        temperature=0.1, 
        top_p=0.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=8192
    )

    # RAG 檢索：改回使用完整文本的前段來抓取歷史範例
    search_query = input_text[:8000]
    retriever = DynamicFewShotRetriever(DB_PATH)
    try:
        few_shot_examples = retriever.get_few_shot_examples(search_query, k=3)
    except Exception as e:
        print(f"     [RAG Error]: {e}")
        few_shot_examples = []

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "=== HISTORICAL REFERENCE CASE ===\n{context_text}\n=== END OF HISTORICAL CASE ==="),
        ("ai", "{answer}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shot_examples,
    )

    print("Stage 1: Entity Extraction (Slicing text into raw samples)...")
    
    stage1_instruction = (
        "You are an expert semiconductor entity extraction agent. "
        "Your ONLY task is to extract the customer metadata and list ALL unique Wafer IDs / Sample IDs present.\n"
        "### CORE RULES:\n"
        "1. VENDOR ROLE: You represent MSS. The 'customer' and 'company' refer to external clients. NEVER list MSS or internal employees.\n"
        "2. LEARNING FROM HISTORY: Look at the provided historical examples to understand what a valid 'Wafer ID' looks like for this specific customer (e.g., alphanumeric codes, format, and where they typically appear in tables or text). "
        "However, DO NOT extract Wafer IDs from the historical cases.\n"
        "3. ANTI-LOOP CRITICAL: DO NOT duplicate Wafer IDs. Once you have listed all unique IDs, you MUST stop generating. Do not repeat the same ID."
    )

    stage1_prompt = ChatPromptTemplate.from_messages([
        ("system", stage1_instruction),
        ("human", "Here are historical examples. Pay close attention to how Wafer IDs are formatted and extracted in these cases:\n<historical_examples>"),
        few_shot_prompt,
        ("human", "</historical_examples>\n\nNow, identify the entities strictly from the following new case:\n\n<target_case>\n{input}\n</target_case>")
    ])

    stage1_chain = stage1_prompt | llm.with_structured_output(Stage1Order)

    try:
        stage1_result = stage1_chain.invoke({"input": input_text})
        if not stage1_result or not stage1_result.samples:
            print("  [Critical] Stage 1 failed to return valid samples.")
            return None, None
        print(f"  -> Extracted {len(stage1_result.samples)} raw samples.")
    except Exception as e:
        print(f"Stage 1 Extraction Failed: {e}")
        return None, None

    # ==========================================
    # 階段二：邏輯推論 (Logic Inference)
    # ==========================================
    print("\nStage 2: Logic Inference (Processing each sample individually)...")
    
    stage2_instruction = (
        "You are an expert semiconductor order logic inference agent. "
        "Your task is to analyze the ENTIRE provided case text, but extract the exact technical parameters ONLY for the specific TARGET WAFER ID requested.\n\n"
        "### HOW TO BUILD THE 'PREPARE' FIELD:\n"
        "- Explicit Parameters: Use exact parameters if present (e.g., 'ALD(W2-A)').\n"
        "- **Explicit Parameters:** If the current case PPT/email provides exact parameters (e.g., 'ALD([Specific Cycle])', '[Specific Material]-bond([Ratio])'), use them exactly as written in the target text.\n"
        "- **Vague Parameters (History Fallback):** If the email uses vague terms like 'ALD Coating', 'HfO2', 'epoxy', or 'Glue', you MUST look at how the 'HISTORICAL REFERENCE CASES' handled similar macros. \n"
        "- Extra steps: Add '+ Top view', '+ DB', or '+ Probing' if explicitly requested.\n\n"
        "### HOW TO BUILD THE 'ROUTE' FIELD:\n"
        "- Select the exact matching standard process route code from the internal list."
    )

    final_samples = []
    lot_base_name = os.path.basename(lot_directory)

    stage2_prompt = ChatPromptTemplate.from_messages([
        ("system", stage2_instruction),
        ("human", "Here are some reference examples from the database. Pay attention to how they map instructions to samples, but DO NOT EXTRACT their Wafer IDs:\n<historical_examples>"),
        few_shot_prompt, 
        ("human", "</historical_examples>\n\n"
                  "Now, read the entire case and infer the parameters STRICTLY for the TARGET WAFER ID.\n\n"
                  "TARGET WAFER ID: {wafer_id}\n\n"
                  "=== FULL CASE CONTEXT ===\n"
                  "{full_text}\n"
                  "=== END OF CASE CONTEXT ===")
    ])

    stage2_chain = stage2_prompt | llm.with_structured_output(Stage2Inference)

    for idx, raw_sample in enumerate(stage1_result.samples):
        if not raw_sample.wafer_id:
            continue
            
        print(f"  -> Inferring parameters for Wafer ID: {raw_sample.wafer_id}")

        try:
            inference_result = stage2_chain.invoke({
                "wafer_id": raw_sample.wafer_id,
                "full_text": input_text
            })
            
            suffix = f"{idx + 1:03d}"
            final_sample = SampleInfo(
                lot_id=f"{lot_base_name}-{suffix}",
                wafer_id=raw_sample.wafer_id,
                thought_process=inference_result.thought_process,
                route=inference_result.route,
                prepare=inference_result.prepare,
                loctestkey=inference_result.loctestkey
            )
            final_samples.append(final_sample)
            
        except Exception as e:
            print(f"     [Inference Error] Failed for {raw_sample.wafer_id}: {e}")

    # ==========================================
    # 組合最終輸出 (Combine Final Result)
    # ==========================================
    final_order = OrderInfo(
        global_analysis=stage1_result.global_analysis,
        company=stage1_result.company,
        customer_name=stage1_result.customer_name,
        samples=final_samples
    )

    print(f"\n=== Final Extraction Result (Processed {len(final_samples)} samples) ===")
    print(final_order.model_dump_json(indent=2))
    
    return final_order, few_shot_examples


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
                res, debug_info = process_lot_request(case_dir, lot_id_override=lot_id_for_debug)
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
                        "ground_truth_matches": gt_matches,
                        "rag_few_shot_examples": debug_info
                    }
                    results.append((case_dir, json.dumps(combined, indent=2, ensure_ascii=False)))
            except Exception as e:
                results.append((case_dir, f"<Processing error: {e}>"))
        else:
            results.append((case_dir, "<Case folder not found>"))

    # Save aggregated results to a timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"data/output/prediction_results/split_results_{ts}.txt"
    with open(out_fname, "w", encoding="utf-8") as outf:
        for case_dir, content in results:
            outf.write(f"=== CASE: {case_dir} ===\n")
            outf.write(content)
            outf.write("\n\n")

    print(f"Batch run complete. Results saved to {out_fname}")