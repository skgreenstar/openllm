#검색/QA 체인 실행 함수

from langchain.chains import RetrievalQA

def run_qa(pipe, vector_db, query):
    """
    Qdrant 벡터 DB로부터 관련 청크를 검색 후
    Mistral-7B 파이프라인(pipe)으로 QA를 수행.
    """
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=pipe,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]
