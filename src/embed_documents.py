#임베딩 생성 및 Qdrant 저장 함수

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

def embed_and_store(docs):
    """
    문서 리스트(docs)를 임베딩한 뒤,
    Qdrant 벡터 DB에 저장한 후 해당 VectorStore 객체를 반환
    """
    # Sentence Transformers 기반 임베딩 모델
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Qdrant에 임베딩된 문서 저장
    vector_db = Qdrant.from_documents(
        docs,
        embedding_model,
        location=":memory:",      # 인메모리 DB 사용 (테스트용)
        collection_name="documents"
    )
    return vector_db
