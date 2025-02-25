#문서 로더/청킹 함수

from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_documents(doc_paths):
    """
    주어진 경로 리스트(doc_paths)에 대해
    1) PDF/Word 문서를 로드
    2) 청크(Chunk) 단위로 분할
    3) 분할된 문서 리스트를 반환
    """
    docs = []
    for path in doc_paths:
        # PDF
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        # Word
        elif path.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
            docs.extend(loader.load())
        else:
            print(f"지원되지 않는 파일 형식입니다: {path}")
    
    # 청킹(Chunking) 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)

    return split_docs
