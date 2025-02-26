from src.load_model import load_mistral_model

def infer_text(prompt, max_length=100, temperature=0.7):
    """
    주어진 프롬프트에 대해 텍스트 생성을 수행합니다.
    
    Args:
        prompt (str): 입력 프롬프트
        max_length (int, optional): 생성될 텍스트의 최대 길이 (기본값: 100)
        temperature (float, optional): 생성 시 샘플링 온도 (기본값: 0.7)
    
    Returns:
        str: 생성된 텍스트
    """
    # LoRA 및 양자화 적용된 모델 로드 (GPU/CPU 환경에 따라 자동 처리)
    llm = load_mistral_model()
    
    # 모델 추론: HuggingFacePipeline 래핑 객체는 callable하므로 직접 호출 가능
    response = llm(prompt, max_length=max_length, do_sample=True, temperature=temperature)
    return response

if __name__ == "__main__":
    # 테스트용 예시 프롬프트
    prompt = "부동산 시장 전망에 대해 알려줘."
    generated_text = infer_text(prompt, max_length=150, temperature=0.8)
    print("생성된 텍스트:", generated_text)

