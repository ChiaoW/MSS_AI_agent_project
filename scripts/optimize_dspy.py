import os
import dspy
import random
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
    os.makedirs("logs", exist_ok=True)
    log_filename = datetime.now().strftime("logs/optimize_metric_%Y%m%d_%H%M%S.log")
    
    metric_logger = logging.getLogger("metric_logger")
    metric_logger.setLevel(logging.INFO)
    
    # 避免重複寫入
    if not metric_logger.handlers:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(file_handler)
        
    return log_filename

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
        print(f"找到快取檔案，直接載入資料集: {cache_path}")
        with open(cache_path, "rb") as f:
            combined_dataset = pickle.load(f)
        print(f"快取載入完成。總計: {len(combined_dataset)} 筆")
    else:
        print("未找到快取檔案，開始從原始文本載入")
        
        gt_path_1 = "data/reference/ground_truth/ground_truth_with_wafer_id_processed.json"
        raw_dir_1 = "data/raw/all_cases"
        dataset_1 = load_dspy_dataset(gt_path_1, raw_dir_1)
        
        gt_path_2 = "data/reference/ground_truth/Kang_Yi_Lin_case_ground_truth_translated.json"
        raw_dir_2 = "data/raw/Kang_Yi_Lin_Merged"
        dataset_2 = load_dspy_dataset(gt_path_2, raw_dir_2)
        
        combined_dataset = dataset_1 + dataset_2
        
        if not combined_dataset:
            print("資料集為空，請檢查路徑。")
            return
            
        # 儲存為快取檔案
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(combined_dataset, f)
        print(f"資料集已存入快取: {cache_path}")

    # 以下邏輯維持不變
    random.seed(42)
    random.shuffle(combined_dataset)

    total_size = len(combined_dataset)
    train_size = int(total_size * 0.6)
    
    trainset = combined_dataset[:train_size]
    devset = combined_dataset[train_size:]
    print(f"資料集切分完成。總計: {total_size} 筆 -> 訓練集: {len(trainset)} 筆, 驗證集: {len(devset)} 筆")

    # 3. 測試 Baseline (未優化)
    evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    uncompiled_extractor = SemiconductorExtractor()
    
    print("\n=== 測試未優化的模型 (Baseline) ===")
    baseline_score = evaluator(uncompiled_extractor, metric=extraction_metric)
    print(f"Baseline 評估分數: {baseline_score}")

    # 4. 執行 BootstrapFewShot Optimizer
    print("\n=== 開始編譯與優化模型 (Optimizer) ===")
    optimizer = BootstrapFewShot(
        metric=extraction_metric,
        max_bootstrapped_demos=3, # 將表現最好的 3 個成功軌跡加入 Prompt
        max_labeled_demos=0       # 不直接使用訓練集原始資料，確保只用模型自己生成的成功軌跡
    )

    # Compile 會花費較多時間，因為它會逐一執行 trainset 並用 metric 驗證
    compiled_extractor = optimizer.compile(
        student=uncompiled_extractor,
        trainset=trainset
    )

    # 5. 測試 Compiled (已優化)
    print("\n=== 測試優化後的模型 (Compiled) ===")
    compiled_score = evaluator(compiled_extractor, metric=extraction_metric)
    print(f"Compiled 評估分數: {compiled_score}")

    # 6. 儲存模型供 dspy_main.py 讀取
    out_dir = "data/output"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/optimized_extractor.json"
    
    compiled_extractor.save(save_path)
    print(f"\n編譯完成！優化後的模型已儲存至: {save_path}")

if __name__ == "__main__":
    main()