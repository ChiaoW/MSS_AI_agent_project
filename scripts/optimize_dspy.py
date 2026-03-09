import os
import dspy
import random
import json
import pickle
import logging
from datetime import datetime
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# 引入自定義模組
from src.dspy_modules import CustomMssRM, SemiconductorExtractor
from src.dspy_utils import load_dspy_dataset, extraction_metric

def setup_metric_logger():
    """設定將 Metric 比對過程存入 log 檔的機制"""
    os.makedirs("data/output/logs", exist_ok=True)
    log_filename = datetime.now().strftime("data/output/logs/optimize_metric_%Y%m%d_%H%M%S.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler() # 保留終端機輸出以便即時查看進度
        ]
    )
    
    # 2. 設定 Metric 專用 Logger (維持原本獨立輸出的設計，不混入主流程日誌)
    metric_log_filename = datetime.now().strftime("data/output/logs/optimize_metric_%Y%m%d_%H%M%S.log")
    metric_logger = logging.getLogger("metric_logger")
    metric_logger.setLevel(logging.INFO)
    
    # 避免重複寫入
    if not metric_logger.handlers:
        file_handler = logging.FileHandler(metric_log_filename, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(file_handler)
        metric_logger.propagate = False # 避免 metric log 跑到 Root Logger 裡
        
    return log_filename, metric_log_filename

def main():
    log_file = setup_metric_logger()

    # 1. 環境設定 (必須與 dspy_main.py 相同)
    DB_URL = "http://localhost:6333"
    LLM_API_BASE = "http://localhost:8000/v1"
    MODEL_NAME = "hosted_vllm/openai/gpt-oss-20b"

    llm = dspy.LM(
        model=MODEL_NAME,
        api_base=LLM_API_BASE,
        api_key="EMPTY",  
        temperature=0,  
        max_tokens=4096,
        cache=False
    )
    my_retriever = CustomMssRM(db_url=DB_URL, k=3)
    dspy.settings.configure(lm=llm, rm=my_retriever)

    cache_path = "data/output/dataset_cache.pkl"

    if os.path.exists(cache_path):
        logging.info(f"找到快取檔案，直接載入資料集: {cache_path}")
        with open(cache_path, "rb") as f:
            combined_dataset = pickle.load(f)
        logging.info(f"快取載入完成。總計: {len(combined_dataset)} 筆")
    else:
        logging.info("未找到快取檔案，開始從原始文本載入")
        
        gt_path_1 = "data/reference/ground_truth/ground_truth_with_wafer_id_processed.json"
        raw_dir_1 = "data/raw/all_cases"
        dataset_1 = load_dspy_dataset(gt_path_1, raw_dir_1)
        
        gt_path_2 = "data/reference/ground_truth/Kang_Yi_Lin_case_ground_truth_translated.json"
        raw_dir_2 = "data/raw/Kang_Yi_Lin_Merged"
        dataset_2 = load_dspy_dataset(gt_path_2, raw_dir_2)
        
        combined_dataset = dataset_1 + dataset_2
        
        if not combined_dataset:
            logging.warning("資料集為空，請檢查路徑。")
            return
            
        # 儲存為快取檔案
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(combined_dataset, f)
        logging.info(f"資料集已存入快取: {cache_path}")

    random.seed(42)
    random.shuffle(combined_dataset)

    total_size = len(combined_dataset)
    train_size = int(total_size * 0.6)
    
    trainset = combined_dataset[:train_size]
    devset = combined_dataset[train_size:]
    logging.info(f"資料集切分完成。總計: {total_size} 筆 -> 訓練集: {len(trainset)} 筆, 驗證集: {len(devset)} 筆")

    # 3. 測試 Baseline (未優化)
    evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    uncompiled_extractor = SemiconductorExtractor()
    
    logging.info("\n=== 測試未優化的模型 (Baseline) ===")
    baseline_score = evaluator(uncompiled_extractor, metric=extraction_metric)
    logging.info(f"Baseline 評估分數: {baseline_score}")

    # 4. 執行 BootstrapFewShot Optimizer
    logging.info("\n=== 開始編譯與優化模型 (Optimizer) ===")
    optimizer = BootstrapFewShot(
        metric=extraction_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=1
    )

    # Compile 會花費較多時間，因為它會逐一執行 trainset 並用 metric 驗證
    compiled_extractor = optimizer.compile(
        student=uncompiled_extractor,
        trainset=trainset
    )
    compiled_extractor.save("data/output/dspy/compiled_mss_extractor.json")

    # 5. 測試 Compiled (已優化)
    logging.info("\n=== 測試優化後的模型 (Compiled) ===")
    compiled_score = evaluator(compiled_extractor, metric=extraction_metric)
    logging.info(f"Compiled 評估分數: {compiled_score}")

    # 6. 儲存模型供 dspy_main.py 讀取
    out_dir = "data/output"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/optimized_extractor.json"
    
    compiled_extractor.save(save_path)
    logging.info(f"\n編譯完成！優化後的模型已儲存至: {save_path}")

    logging.info("\n=== 顯示最後 3 次 LLM 互動歷史於終端機 ===")
    llm.inspect_history(n=3)

    history_out_dir = "data/output/debug_prompt"
    os.makedirs(history_out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f"{history_out_dir}/optimizer_history_{ts}.txt"

    with open(history_file, "w", encoding="utf-8") as f:
        f.write("=== Optimizer Training LLM History ===\n")
        f.write(f"Total LLM Calls during Optimization: {len(llm.history)}\n\n")
        
        for i, interaction in enumerate(llm.history):
            f.write(f"\n{'='*20} LLM CALL #{i+1} {'='*20}\n")
            
            # 寫入傳送給 LLM 的 Prompt
            f.write(">>> EXACT PROMPT SENT TO LLM:\n")
            prompt = interaction.get('prompt')
            if prompt is None:
                kwargs = interaction.get('kwargs', {})
                messages = kwargs.get('messages')
                if messages:
                    prompt = json.dumps(messages, indent=2, ensure_ascii=False)
                else:
                    prompt = "N/A (Prompt data not found in history)"
            f.write(str(prompt) + "\n\n")
            
            # 寫入 LLM 的原始回覆
            f.write("<<< LLM RAW RESPONSE:\n")
            response_obj = interaction.get('response', '')
            if isinstance(response_obj, list) and len(response_obj) > 0:
                if hasattr(response_obj[0], 'message'):
                    f.write(str(response_obj[0].message.content) + "\n")
                else:
                    f.write(str(response_obj) + "\n")
            else:
                f.write(str(response_obj) + "\n")
            f.write("\n\n")
            
    logging.info(f"完整的 Optimizer 互動歷史已儲存至: {history_file}")

if __name__ == "__main__":
    main()