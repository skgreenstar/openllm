#Mistral 모델 로딩 함수

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def load_mistral_model(model_name="mistralai/Mistral-7B-v0.3"):
    """
    Mistral-7B 모델을 로딩하여 pipeline 객체로 반환.
    device_map="auto" 설정 시 GPU가 있으면 자동 할당.
    """
    pipe = pipeline("text-generation", model=model_name, device_map="auto", trust_remote_code=True)
    

    # HuggingFacePipeline으로 래핑하여 LangChain의 LLM 인터페이스에 맞춥니다.
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

