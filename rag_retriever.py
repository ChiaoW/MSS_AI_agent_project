from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse
# from langchain_openai import OpenAIEmbeddings 
from qdrant_client import QdrantClient
from typing import List, Dict
import torch
from langchain_huggingface import HuggingFaceEmbeddings

class DynamicFewShotRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.col_english = "semiconductor_orders_english"
        self.col_original = "semiconductor_orders"
        
        # 必須與 build_vector_db 使用相同的 Embedding 模型
        self.embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # 初始化兩個 Vector Store
        self.client = QdrantClient(path=self.db_path)
        
        # [CRITICAL FIX] 2. 將 client 傳入 VectorStore，而不是傳 path
        # 注意：這裡改用標準建構子，而不是 from_existing_collection (比較好控制 client)
        self.vs_english = QdrantVectorStore(
            client=self.client, 
            collection_name=self.col_english,
            embedding=self.embeddings,
            # sparse_embedding=self.sparse_embeddings,
            # retrieval_mode="hybrid"
        )
        
        self.vs_original = QdrantVectorStore(
            client=self.client,
            collection_name=self.col_original,
            embedding=self.embeddings,
            # sparse_embedding=self.sparse_embeddings,
            # retrieval_mode="hybrid"
        )

    def get_few_shot_examples(self, query_text: str, k: int = 5) -> List[Dict]:
        retriever_org = self.vs_original.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
        
        retriever_eng = self.vs_english.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
        docs_org = retriever_org.invoke(query_text)
        docs_eng = retriever_eng.invoke(query_text)

        selected_docs = []
        
        num_org = max(1, k-2) 
        selected_docs.extend(docs_org[:num_org])
        
        remaining = k - len(selected_docs)
        if remaining > 0:
            existing_ids = {d.metadata.get("lot_id") for d in selected_docs}
            
            for doc in docs_eng:
                if len(selected_docs) >= k:
                    break
                if doc.metadata.get("lot_id") not in existing_ids:
                    selected_docs.append(doc)

            if len(selected_docs) < k:
                needed = k - len(selected_docs)
                start_idx = num_org
                selected_docs.extend(docs_eng[start_idx : start_idx + needed])
        examples = []
        for doc in selected_docs:
            full_text = doc.page_content

            limit = 12000 
            if len(full_text) > limit:
                truncated_content = full_text[:limit] + "\n...[Content Truncated]..."
            else:
                truncated_content = full_text

            # is_translated = any(d.page_content == doc.page_content for d in docs_eng)
            # source_tag = "[Translated DB]" if is_translated else "[Original DB (Real Case)]"
            
            examples.append({
                "context_text": f"{truncated_content}", 
                "answer": doc.metadata["output_json"]
            })
            
        return examples