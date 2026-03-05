import os
import json
from datetime import datetime
import dspy

# 引入你原本寫好的模組
from src.file_processor import UniversalFileProcessor
from src.pydantic_schema import OrderInfo, SampleInfo, Stage1Order, Stage2Inference, Stage1Sample
from src.rag_retriever import DynamicFewShotRetriever
from src.dspy_modules import CustomMssRM, Stage1Signature, Stage2Signature, SemiconductorExtractor, save_debug_prompt

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
    temperature=0.1,  
    max_tokens=8192,
    cache=False
)

# 註冊我們客製化的 Retriever
my_retriever = CustomMssRM(db_path=DB_URL, k=3)
dspy.settings.configure(lm=llm, rm=my_retriever)


# ==========================================
# 4. 主程式執行區塊 (讀檔與批次處理)
# ==========================================
def truncate_text(text: str, max_chars: int = 40000) -> str:
    if len(text) <= max_chars:
        return text
    print('Text Truncated')
    return text[:max_chars] + "\n...[Content Truncated due to length limit]..."

def process_lot_request(lot_directory: str):
    processor = UniversalFileProcessor()
    print(f"\nProcessing files in {lot_directory}...")
    try:
        full_context_text = processor.process_directory(lot_directory)
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

    input_text = truncate_text(full_context_text)
    lot_base_name = os.path.basename(lot_directory)

    # 實例化我們的 DSPy 模組
    extractor = SemiconductorExtractor()
    
    try:
        prediction = extractor(input_text=input_text, lot_base_name=lot_base_name)
    except Exception as e:
        print(f"DSPy Extraction Exception: {e}")
        prediction = None
    finally:
        save_debug_prompt(
            lot_id=lot_base_name, 
            llm_instance=llm, 
            input_text=extractor.latest_input, 
            historical_context=extractor.latest_context
        )

    if prediction is None:
         print("DSPy Extractor returned None (Likely Stage 1 Failed).")
         return None, extractor.latest_context

    return prediction.final_order, prediction.historical_context



if __name__ == "__main__":
    # 測試用的案例資料夾
    cases = [
        "data/raw/all_cases/T26030201"
        # "data/raw/Kang_Yi_Lin_Merged/T25082545",
    ]

    results = []

    for case_dir in cases:
        if os.path.exists(case_dir):
            print(f"\n=== Processing case: {case_dir} ===")
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
                print(f"=== Final DSPy Extraction Result ===")
                print(pred_json)
        else:
            print(f"Case folder not found: {case_dir}")

    # 將結果存檔
    os.makedirs("data/output/prediction_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"data/output/prediction_results/dspy_results_{ts}.txt"
    
    with open(out_fname, "w", encoding="utf-8") as outf:
        for case_dir, content in results:
            outf.write(f"=== CASE: {case_dir} ===\n")
            outf.write(content)
            outf.write("\n\n")

    print(f"\nBatch run complete. DSPy Results saved to {out_fname}")