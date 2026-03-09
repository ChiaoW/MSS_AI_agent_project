import re
import json
import dspy
import logging

def normalize_text(text):
    """基礎的字串常規化：轉小寫、去除多餘空白"""
    if text is None or str(text).lower() == "nan":
        return ""
    return str(text).lower().strip()

def normalize_prepare_steps(prepare_str):
    """
    針對 prepare 欄位進行深度常規化
    1. 轉小寫
    2. 以 '+' 切割步驟，並自動吸收 '+' 前後的任何空白
    3. 針對特定同義詞進行統一替換
    4. 回傳 Set (無序集合)，以忽略步驟先後順序的差異
    """
    if not prepare_str:
        return set()
        
    # 統一轉小寫，並使用正規表達式以 '+' 切割，同時移除多餘空白
    raw_steps = re.split(r'\s*\+\s*', normalize_text(prepare_str))
    
    normalized_steps = set()
    for step in raw_steps:
        if not step:
            continue
            
        # 規則 A: 處理 DB 定位相關同義詞
        if step in ["db positioning", "positioning", "db-positioning"]:
            step = "db"
            
        # 規則 B: 處理 M-bond 溫度與時間變體 
        # 捕捉包含 m-bond 且帶有 60 與 30 數字的任何寫法
        elif "m-bond" in step and "60" in step and "30" in step:
            step = "m-bond(60/30)"
            
        # 規則 C: 處理 Pi-bond 溫度與時間變體
        elif "pi-bond" in step and "60" in step and "30" in step:
            step = "pi-bond(60/30)"
        
        # 規則 D: 處理 Top view 變體 (將 topview, top-view, top view 統一收斂為 top view)
        elif step.replace(" ", "").replace("-", "") == "topview":
            step = "top view"
            
        normalized_steps.add(step)
        
    return normalized_steps

def normalize_loctestkey(loc_str):
    """針對 loctestkey 進行常規化處理"""
    if not loc_str:
        return ""
        
    loc = normalize_text(loc_str)
    
    # 處理 X-cut / Y-cut 變體 (例如: 23p-xcut, 23P y-cut 統一為 23p x-cut / 23p y-cut)
    # 步驟 1：將 x/y 與 cut 之間的任何空白或減號統一為單一減號
    loc = re.sub(r'([xy])[\s\-]*cut', r'\1-cut', loc)
    # 步驟 2：將前方字元與 x-cut/y-cut 之間的減號替換為空白
    loc = re.sub(r'([a-z0-9])\-([xy]\-cut)', r'\1 \2', loc)
    
    return loc

def extraction_metric(example, pred, trace=None):
    if pred is None or pred.final_order is None:
        return 0.0
        
    expected_samples = example.expected_samples
    predicted_samples = pred.final_order.samples
    
    if not predicted_samples:
        return 0.0
        
    pred_dict = {str(s.wafer_id).lower(): s for s in predicted_samples}
    
    score = 0.0
    total_attributes = len(expected_samples) * 4 # 每個 sample 評估 wafer_id, route, prepare, loctestkey

    logger = logging.getLogger("metric_logger")
    
    logger.info(f"\n[Metric Debug] 評估 Lot: {example.lot_base_name}")
    
    for exp_s in expected_samples:
        wafer_id = str(exp_s["wafer_id"]).lower()
        if wafer_id in pred_dict:
            score += 1.0 # Wafer ID 命中
            
            p_sample = pred_dict[wafer_id]
            p_route = normalize_text(p_sample.route)
            e_route = normalize_text(exp_s.get("route"))
            p_prepare = normalize_prepare_steps(p_sample.prepare)
            e_prepare = normalize_prepare_steps(exp_s.get("prepare"))
            p_loc = normalize_loctestkey(p_sample.loctestkey)
            e_loc = normalize_loctestkey(exp_s.get("loctestkey"))
            
            # [修改] 使用 logger.info 紀錄比對細節
            logger.info(f"  -> Wafer: {wafer_id}")
            logger.info(f"     [Route]   預測: {p_route:<20} | 答案: {e_route}")
            logger.info(f"     [Prepare] 預測: {str(p_prepare):<20} | 答案: {str(e_prepare)}")
            logger.info(f"     [LocTest] 預測: {p_loc:<20} | 答案: {e_loc}")
            
            if p_route == e_route:
                score += 1.0
            if p_prepare == e_prepare:
                score += 1.0
            if p_loc == e_loc:
                score += 1.0
        else:
            logger.info(f"  -> Wafer: {wafer_id} [未命中/漏抓]")
                
    final_score = score / total_attributes if total_attributes > 0 else 0.0
    logger.info(f"  => 此 Lot 得分: {final_score:.2f}")
    
    return final_score

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
            input_text = full_text[:20000] # 依照你原本的截斷邏輯
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