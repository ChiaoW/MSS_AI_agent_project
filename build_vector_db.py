import json
import os
import torch
from typing import List, Dict
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings

# 選擇 Embedding 模型
# 選項 A: 使用本地 FastEmbed (推薦，無需依賴 vLLM server 的 embedding endpoint)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse
# 選項 B: 使用 vLLM 的 OpenAI 兼容接口 (若 vLLM 啟動參數包含 --embedding)
from langchain_openai import OpenAIEmbeddings

from file_processor import UniversalFileProcessor

# 設定路徑
GT_JSON_PATH = "data/reference/ground_truth/ground_truth_with_wafer_id_processed.json"
CASES_DIR = "data/raw/all_cases_in_ground_truth"
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "semiconductor_orders"

def load_ground_truth_data(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_database():
    print("Step 1: Loading Ground Truth Data...")
    gt_data = load_ground_truth_data(GT_JSON_PATH)
    
    documents = []
    processor = UniversalFileProcessor()
    
    print("Step 2: Processing Raw Files for Knowledge Base...")
    # gt_data 是個 Dict，Key 是 Lot ID (e.g., "T25111801")
    for lot_id, data in gt_data.items():
        lot_path = os.path.join(CASES_DIR, lot_id)
        
        # 1. 獲取 Input Text (原始檔案內容)
        if os.path.exists(lot_path):
            raw_text = processor.process_directory(lot_path)
        else:
            print(f"Warning: Raw files for {lot_id} not found at {lot_path}. Using placeholder.")
            # 為了讓程式能跑，若無檔案則生成合成文字 (實際運作請確保檔案存在)
            raw_text = f"Simulated email content for Lot {lot_id}. Customer requests analysis for {len(data['samples'])} wafers."

        # 2. 獲取 Output JSON (作為 Few-Shot 的 Answer)
        # 將該 Lot 的整包 Output 轉為 JSON String
        output_json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        # 3. 提取 Metadata (Company, Route 統計)
        # 從 samples 中統計出現最多次的 Company 和 Route 標籤作為索引
        companies = [s.get('company') for s in data.get('samples', []) if s.get('company')]
        routes = [s.get('route') for s in data.get('samples', []) if s.get('route')]
        
        primary_company = max(set(companies), key=companies.count) if companies else "Unknown"
        primary_route = max(set(routes), key=routes.count) if routes else "Unknown"

        # 4. 建立 LangChain Document
        # page_content 放 "Raw Text" (用來被搜尋)
        # metadata 放 "Output JSON" (搜尋到後用來當範例)
        doc = Document(
            page_content=raw_text,
            metadata={
                "lot_id": lot_id,
                "output_json": output_json_str, # 這就是 RAG 要 Retrieve 的 "Answer"
                "company": primary_company,
                "route_tag": primary_route
            }
        )
        documents.append(doc)

    print(f"Step 3: Ingesting {len(documents)} documents into Qdrant...")

    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    QdrantVectorStore.from_documents(
            documents,
            embeddings,
            sparse_embedding=sparse_embeddings,
            path=DB_PATH,
            collection_name=COLLECTION_NAME,
            force_recreate=True
    )
        
    print("SUCCESS: Vector Database built successfully!")

if __name__ == "__main__":
    build_database()