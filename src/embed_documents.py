#임베딩 생성 및 Qdrant 저장 함수
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
#from sentence_transformers import SentenceTransformer

def embed_and_store(docs):
    """
    문서 리스트(docs)를 임베딩한 뒤,
    Qdrant 벡터 DB에 저장한 후 해당 VectorStore 객체를 반환
    """
    # Sentence Transformers 기반 임베딩 모델
    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # fine-tuning된 모델 로드
    fine_tuned_model_path = "output/finetuned-model"
    embedding_model = HuggingFaceEmbeddings(model_name=fine_tuned_model_path)

    # 환경 변수(QDRANT_URL)에서 Qdrant 서비스 주소를 가져옵니다.
    # 환경 변수가 없으면 기본값 ":memory:"인메모리 DB를 사용합니다.
    qdrant_url = os.environ.get("QDRANT_URL", ":memory:")
    
    # Qdrant에 임베딩된 문서 저장
    vector_db = Qdrant.from_documents(
        docs,
        embedding_model,
        location=qdrant_url,      # 인메모리 DB 사용 (테스트용)
        collection_name="documents"
    )
    return vector_db
