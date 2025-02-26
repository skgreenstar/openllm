from src.inference import infer_text

def main():
    print("LLM CLI Inference")
    print("종료하려면 'q'를 입력하세요.\n")
    
    while True:
        prompt = input("프롬프트를 입력하세요: ")
        if prompt.lower() == 'q':
            print("종료합니다.")
            break

        # 추론 실행 (원하는 옵션에 맞게 max_length, temperature 조절)
        response = infer_text(prompt, max_length=150, temperature=0.7)
        print("생성된 텍스트:\n", response)
        print("-" * 80)

if __name__ == "__main__":
    main()
