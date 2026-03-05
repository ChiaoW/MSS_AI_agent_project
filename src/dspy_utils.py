import json
import dspy

def extraction_metric(example, pred, trace=None):
    if pred is None or pred.final_order is None:
        return 0.0
        
    expected_samples = example.expected_samples
    predicted_samples = pred.final_order.samples
    
    if not predicted_samples:
        return 0.0
        
    # 將預測結果轉為容易比對的格式
    pred_dict = {s.wafer_id: s for s in predicted_samples}
    
    score = 0.0
    total_attributes = len(expected_samples) * 4 # 每個 sample 評估 4 個欄位
    
    for exp_s in expected_samples:
        wafer_id = exp_s["wafer_id"]
        if wafer_id in pred_dict:
            score += 1.0 # Wafer ID 命中
            
            p_sample = pred_dict[wafer_id]
            # 寬鬆比對或精確比對
            if str(p_sample.route).strip() == str(exp_s.get("route")).strip():
                score += 1.0
            if str(p_sample.prepare).strip() == str(exp_s.get("prepare")).strip():
                score += 1.0
            if str(p_sample.loctestkey).strip() == str(exp_s.get("loctestkey")).strip():
                score += 1.0
                
    # 回傳 0 到 1 之間的分數
    return score / total_attributes if total_attributes > 0 else 0.0

def load_dspy_dataset(ground_truth_path, raw_data_dir):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    dataset = []
    # 這裡你需要實作讀取該 lot_id 對應的文本
    from src.file_processor import UniversalFileProcessor
    processor = UniversalFileProcessor()
    
    for lot_id, data in gt_data.items():
        case_dir = f"{raw_data_dir}/{lot_id}"
        try:
            full_text = processor.process_directory(case_dir)
            input_text = full_text[:40000] # 依照你原本的截斷邏輯
        except Exception:
            continue # 如果找不到檔案則跳過
            
        # 將預期的 samples 轉為字典或 Pydantic 物件以便後續比對
        expected_samples = data["samples"]
        
        # 建立 dspy.Example，並使用 with_inputs 標註哪些欄位是模型的輸入
        example = dspy.Example(
            input_text=input_text,
            lot_base_name=lot_id,
            expected_samples=expected_samples
        ).with_inputs("input_text", "lot_base_name")
        
        dataset.append(example)
        
    return dataset

# 將資料切分為 Train 與 Dev
dataset = load_dspy_dataset("data/reference/ground_truth/ground_truth.json", "data/raw/all_cases")
trainset = dataset[:15]
devset = dataset[15:]