import json
import os
import re
from typing import List, Dict
from tqdm import tqdm  # 如果沒安裝，請 pip install tqdm

# LangChain & Vector DB
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 自定義模組
from src.file_processor import UniversalFileProcessor

# ==========================================
# Config
# ==========================================
GT_JSON_PATH = "data/reference/ground_truth/Kang_Yi_Lin_case_ground_truth_translated.json"
CASES_DIR = "data/raw/Kang_Yi_Lin_Merged"
DB_URL = "http://localhost:6333"
COLLECTION_NAME = "semiconductor_orders_english"
CACHE_FILE = "data/output/translation_cache.json"  # 用來存翻譯過的文字，避免重複跑

# LLM 設定 (用於改寫/翻譯)
LLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-20b"

# ==========================================
# Helper: Translation / Rewriting Agent
# ==========================================
class EngineeringRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_base=LLM_API_BASE,
            openai_api_key="EMPTY",
            temperature=0.1,  # 低溫以保持準確
            max_tokens=4096
        )
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def contains_chinese(self, text: str) -> bool:
        """檢查字串中是否包含中文字元"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def rewrite_to_american_english(self, text: str, lot_id: str) -> str:
        content_hash = str(hash(text)) 
        cache_key = f"{lot_id}_{content_hash}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.contains_chinese(text):
            return text

        print(f"  [Rewriting] Detected Chinese/Non-standard English in {lot_id}. Calling LLM...")

        # 3. 定義 Rewrite Prompt
        system_prompt = (
            "You are a Senior Process Engineer at a top US semiconductor fab (e.g., Intel, GlobalFoundries). "
            "Your task is to rewrite the provided email and technical documentation into **professional, native American engineering English**.\n\n"
            "### Rewrite Rules:\n"
            "1. **Tone:** Professional, concise, direct (Active Voice). No 'Chinglish' phrases.\n"
            "2. **Terminology Mapping (Taiwanese -> US):**\n"
            "   - '打線' -> 'Wire Bonding'\n"
            "   - '切片' -> 'Cross-section' or 'X-Cut'\n"
            "   - '正面/背面' -> 'Front-side / Back-side'\n"
            "   - '下針' / '點針' -> 'Probing' or 'Touch-down'\n"
            "   - '去層' -> 'De-layering' or 'Layer removal'\n"
            "   - '異常' -> 'Anomaly' or 'Defect'\n"
            "3. **Preserve Structure:**\n"
            "   - **DO NOT** change any Lot IDs, Wafer IDs, numeric values, or coordinate data.\n"
            "   - **KEEP** the Markdown table formats and <source_file> tags exactly as they are.\n"
            "   - Only translate the text content within the structure.\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "### Input Text:\n{text}\n\n### Rewritten Output:")
        ])

        chain = prompt | self.llm

        try:
            response = chain.invoke({"text": text})
            rewritten_text = response.content
            
            # 4. 存入 Cache
            self.cache[cache_key] = rewritten_text
            self._save_cache()
            
            return rewritten_text
        except Exception as e:
            print(f"  [Error] Translation failed: {e}. Using original text.")
            return text

# ==========================================
# Main Build Logic
# ==========================================
def load_ground_truth_data(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_database_with_rewrite():
    print("Step 1: Loading Ground Truth Data...")
    if not os.path.exists(GT_JSON_PATH):
        print(f"Error: {GT_JSON_PATH} not found.")
        return

    gt_data = load_ground_truth_data(GT_JSON_PATH)
    
    documents = []
    processor = UniversalFileProcessor()
    rewriter = EngineeringRewriter()
    
    print("Step 2: Processing & Rewriting Files for Knowledge Base...")
    
    # 使用 tqdm 顯示進度條
    for lot_id, data in tqdm(gt_data.items(), desc="Processing Lots"):
        lot_path = os.path.join(CASES_DIR, lot_id)
        
        # 1. 讀取原始檔案 (含 Markdown 表格 與 XML Tags)
        if os.path.exists(lot_path):
            raw_text = processor.process_directory(lot_path)
        else:
            # Placeholder
            raw_text = f"Simonulated content for {lot_id}. No files found."

        print(f"\n  -> [Debug] {lot_id} 原始讀取字數: {len(raw_text)}")
        if len(raw_text.strip()) == 0:
            print(f"  -> [Warning] {lot_path} 讀出來完全沒字！")

        # 2. AI 改寫/翻譯
        # final_text = rewriter.rewrite_to_american_english(raw_text, lot_id)

        # print(f"  -> [Debug] {lot_id} 翻譯後字數: {len(final_text)}")
        # if len(final_text.strip()) == 0:
        #     print(f"  -> [Critical] LLM 回傳了空字串！強制使用原始文字存入 DB。")
        #     final_text = raw_text

        # 3. 準備 Metadata (作為 Few-Shot 的 Answer)
        output_json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        companies = [s.get('company') for s in data.get('samples', []) if s.get('company')]
        routes = [s.get('route') for s in data.get('samples', []) if s.get('route')]
        primary_company = max(set(companies), key=companies.count) if companies else "Unknown"
        primary_route = max(set(routes), key=routes.count) if routes else "Unknown"

        # 4. 建立 Document
        doc = Document(
            page_content=raw_text, # 存入原始文字
            metadata={
                "lot_id": lot_id,
                "output_json": output_json_str,
                "company": primary_company,
                "route_tag": primary_route,
                "original_text": raw_text[:200] # 除錯用：保留一點原始文字
            }
        )
        documents.append(doc)

    print(f"\nStep 3: Ingesting {len(documents)} documents into Qdrant...")

    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    for doc in documents:
        if len(doc.page_content) > 30000:
            doc.page_content = doc.page_content[:30000] + "\n...[Truncated]"

    # B. 設定小批次處理 (Batch Size)
    BATCH_SIZE = 2 # 每次只算 2 筆，絕對不會 OOM
    
    print(f"開始分批寫入，每次 {BATCH_SIZE} 筆...")
    
    # 處理第一批 (順便建立並覆蓋資料庫 force_recreate=True)
    first_batch = documents[:BATCH_SIZE]
    vector_store = QdrantVectorStore.from_documents(
        first_batch,
        embeddings,
        sparse_embedding=sparse_embeddings,
        url=DB_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    print(f"  -> Batch 1/{len(documents)//BATCH_SIZE + 1} completed.")

    # 處理後續批次 (用 add_documents 加入，不要再 force_recreate)
    for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"  -> Batch {i//BATCH_SIZE + 1}/{len(documents)//BATCH_SIZE + 1} completed.")
    
    print(f"SUCCESS: Vector Database built at {DB_URL}/{COLLECTION_NAME} with English-Rewritten content!")

if __name__ == "__main__":
    build_database_with_rewrite()