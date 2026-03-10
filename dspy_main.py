import os
import json
import logging
from datetime import datetime
import dspy

# 引入你原本寫好的模組
from src.file_processor import UniversalFileProcessor
from src.pydantic_schema import OrderInfo, SampleInfo, Stage1Order, Stage2Inference, Stage1Sample
from src.rag_retriever import DynamicFewShotRetriever
from src.dspy_modules import CustomMssRM, Stage1Signature, Stage2Signature, SemiconductorExtractor, save_debug_prompt

os.makedirs("data/output/logs", exist_ok=True)
logging.basicConfig(
    filename="data/output/logs/dspy_main.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# ==========================================
# 1. DSPy 環境與模型設定
# ==========================================
DB_URL = "http://localhost:6333"
LLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "hosted_vllm/openai/gpt-oss-20b"

# 設定 LLM
llm = dspy.LM(
    model=MODEL_NAME,
    api_base=LLM_API_BASE,
    api_key="EMPTY",  
    temperature=0,  
    max_tokens=5000,
    cache=False
)

my_retriever = CustomMssRM(db_url=DB_URL, k=3)
dspy.settings.configure(lm=llm, rm=my_retriever)


# ==========================================
# 4. 主程式執行區塊 (讀檔與批次處理)
# ==========================================
def truncate_text(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    logging.warning("Text Truncated")
    return text[:max_chars] + "\n...[Content Truncated due to length limit]..."

def process_lot_request(lot_directory: str):
    processor = UniversalFileProcessor()
    logging.info(f"Processing files in {lot_directory}...")
    try:
        full_context_text = processor.process_directory(lot_directory)
    except Exception as e:
        logging.error(f"Error reading files: {e}")
        return None, None

    input_text = truncate_text(full_context_text)
    lot_base_name = os.path.basename(lot_directory)

    # 實例化我們的 DSPy 模組
    extractor = SemiconductorExtractor()

    optimized_model_path = "data/output/optimized_extractor.json"
    if os.path.exists(optimized_model_path):
        extractor.load(optimized_model_path)
    
    try:
        prediction = extractor(input_text=input_text, lot_base_name=lot_base_name)
    except Exception as e:
        logging.error(f"DSPy Extraction Exception: {e}")
        prediction = None
    finally:
        save_debug_prompt(
            lot_id=lot_base_name, 
            llm_instance=llm, 
            input_text=extractor.latest_input, 
            historical_context=extractor.latest_context
        )

    if prediction is None:
         logging.warning("DSPy Extractor returned None (Likely Stage 1 Failed).")
         return None, extractor.latest_context

    return prediction.final_order, prediction.historical_context



if __name__ == "__main__":
    # Load test cases from answers_20226.json
    answers_path = "data/reference/answers/answers_2026.json"
    raw_base_path = "data/raw/all_cases"
    cases = []

    if os.path.exists(answers_path):
        try:
            with open(answers_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle Dict format (Key = Lot ID)
            if isinstance(data, dict):
                for lot_id in data.keys():
                    cases.append(os.path.join(raw_base_path, lot_id))
            
            # Handle List format (extract base Lot ID from samples)
            elif isinstance(data, list):
                for item in data:
                    samples = item.get("samples", [])
                    if samples and "lot_id" in samples[0]:
                        # e.g., "T26012301-001" -> "T26012301"
                        full_id = samples[0]["lot_id"]
                        base_id = full_id.split("-")[0]
                        cases.append(os.path.join(raw_base_path, base_id))
        except Exception as e:
            logging.error(f"Error loading answers file: {e}")

    # Deduplicate
    cases = sorted(list(set(cases)))

    results = []

    for case_dir in cases:
        if os.path.exists(case_dir):
            logging.info(f"\n=== Processing case: {case_dir} ===")
            final_order, hist_context = process_lot_request(case_dir)
            
            if final_order is None:
                results.append((case_dir, "<No result or parsing failed>"))
            else:
                pred_json = final_order.model_dump_json(indent=2)
                
                combined = {
                    "prediction": json.loads(pred_json),
                    "rag_context_used": hist_context
                }
                results.append((case_dir, json.dumps(combined, indent=2, ensure_ascii=False)))
                logging.info(f"=== Final DSPy Extraction Result ===")
                logging.info(pred_json)
        else:
            logging.warning(f"Case folder not found: {case_dir}")

    # 將結果存檔
    os.makedirs("data/output/prediction_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"data/output/prediction_results/dspy_results_2026_{ts}.txt"
    
    with open(out_fname, "w", encoding="utf-8") as outf:
        for case_dir, content in results:
            outf.write(f"=== CASE: {case_dir} ===\n")
            outf.write(content)
            outf.write("\n\n")

    logging.info(f"\nBatch run complete. DSPy Results saved to {out_fname}")