import json
import os
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from transformers import AutoTokenizer

# 複用模組
from file_processor import UniversalFileProcessor

# ==========================================
# 1. 配置與路徑
# ==========================================
GT_JSON_PATH = "data/reference/ground_truth/Kang_Yi_Lin_case_ground_truth_translated.json"
CASES_DIR = "data/raw/Kang_Yi_Lin_Merged"
# 使用 Unsloth 優化過的 4-bit 版本
MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
OUTPUT_DIR = "outputs/kang_yi_sft_model"

CACHE_PATH = "data/sft_train_cache.jsonl"

def prepare_sft_dataset(tokenizer):

    if os.path.exists(CACHE_PATH):
        print(f"載入現有快取資料：{CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return Dataset.from_list([json.loads(line) for line in f])
        
    print("正在準備訓練資料...")
    with open(GT_JSON_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    processor = UniversalFileProcessor()
    train_data = []

    system_instruction = (
        "You are an expert semiconductor order extraction agent. "
        "Extract information into the requested JSON schema.\n"
        "VENDOR ROLE: You represent MSS. The 'customer' is the external client.\n"
    )

    for lot_id, data in tqdm(gt_data.items(), desc="讀取原始檔案"):
        lot_path = os.path.join(CASES_DIR, lot_id)
        if os.path.exists(lot_path):
            raw_text = processor.process_directory(lot_path)
            
            # 使用更標準的對話格式
            message = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Please extract JSON for: {raw_text}"},
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False, indent=2)}
            ]
            
            # 透過 tokenizer 轉換成模型理解的字串並加上 EOS
            formatted_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            train_data.append({"text": formatted_text})

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return Dataset.from_list(train_data)

def run_finetuning():
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = prepare_sft_dataset(temp_tokenizer)
    
    input("資料準備完成！請現在關閉所有 vLLM Docker 容器以釋放顯存，然後按 Enter 繼續訓練...")

    # A. 載入模型 (Blackwell 優化)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 4096,
        dtype = torch.bfloat16, # Blackwell 核心支援
        load_in_4bit = True, 
    )

    # B. 設定 LoRA (針對 GPT-OSS 的 MoE 架構優化)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
    )

    # C. 設定訓練器
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = prepare_sft_dataset(tokenizer),
        dataset_text_field = "text",
        max_seq_length = 4096,
        args = TrainingArguments(
            per_device_train_batch_size = 4, 
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, 
            learning_rate = 5e-5,
            bf16 = True, # Blackwell 強烈建議
            logging_steps = 1,
            optim = "adamw_8bit",
            output_dir = OUTPUT_DIR,
        ),
    )

    print("開始微調訓練...")
    trainer.train()

    # D. 儲存 LoRA (備份)
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    
    # E. [重要] 合併並儲存為 16bit 模型，供 vLLM 載入
    print("正在合併模型以供 vLLM 使用...")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged_model", 
        tokenizer, 
        save_method = "merged_16bit"
    )
    print(f"完成！請將 {OUTPUT_DIR}/merged_model 資料夾移至 Docker 卷中。")

if __name__ == "__main__":
    run_finetuning()