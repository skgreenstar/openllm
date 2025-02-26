import gradio as gr
from src.inference import infer_text

def generate_text(prompt, max_length, temperature):
    """
    UI에서 입력받은 프롬프트와 옵션으로 텍스트 생성을 수행합니다.
    """
    return infer_text(prompt, max_length=int(max_length), temperature=float(temperature))

# Gradio 인터페이스 구성
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=4, placeholder="프롬프트를 입력하세요...", label="프롬프트"),
        gr.Slider(minimum=10, maximum=500, step=10, value=100, label="최대 길이"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="온도")
    ],
    outputs=gr.Textbox(label="생성된 텍스트"),
    title="LLM 프롬프트 인터페이스",
    description="프롬프트를 입력하고 'Submit'을 클릭하면 모델이 응답을 생성합니다."
)

if __name__ == "__main__":
    iface.launch()
