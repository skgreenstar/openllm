import os
import logging
from src.load_model import load_mistral_model
from src.parse_documents import parse_documents
from src.embed_documents import embed_and_store
from src.run_qa import run_qa
from src.load_model import load_mistral_model

# 로그 설정: INFO 레벨로 출력, 간단한 시간, 레벨, 메시지 형식 적용
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("1) Mistral-7B 모델 로드 시작")
    llm = load_mistral_model()
    logging.info("Mistral-7B 모델 로드 완료")

    logging.info("2) 문서 파싱 시작")
    doc_paths = [
        "data/sample.pdf",
        #"data/sample.docx"
    ]
    docs = parse_documents(doc_paths)
    logging.info(f"문서 파싱 완료 - 총 {len(docs)} 개의 청크 생성됨")

    logging.info("3) 문서 임베딩 및 Qdrant 저장 시작")
    vector_db = embed_and_store(docs)
    logging.info("문서 임베딩 및 Qdrant 저장 완료")

    logging.info("4) QA 질의 실행 시작")
    query = "이 문서에서 중요한 개념 3가지를 알려줘."
    answer, sources = run_qa(llm, vector_db, query)
    logging.info("QA 질의 실행 완료")

    logging.info("5) 결과 출력")
    print("🔎 검색된 문서에서 추출한 답변:")
    print(answer)

    print("\n📄 사용된 문서 출처:")
    for doc in sources:
        print(doc.metadata)

if __name__ == "__main__":
    main()

