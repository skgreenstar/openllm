#Mistral 모델 로딩 함수
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain.llms import HuggingFacePipeline

def load_mistral_model(base_model_name="mistralai/Mistral-7B-v0.3", lora_path="output/finetuned_lora"):
    """
    Mistral-7B 모델을 로드하고 LoRA 어댑터를 적용한 후,
    LangChain에서 사용할 수 있도록 HuggingFacePipeline으로 래핑하여 반환합니다.

    - GPU 환경이면 BitsAndBytesConfig를 통해 8비트 양자화를 적용하여 모델 로드(PTQ 적용),
    - CPU 환경이면 일반 모델을 로드한 뒤, torch.quantization.quantize_dynamic을 사용하여 동적 양자화를 적용
    """

    if torch.cuda.is_available():
        # GPU: BitsAndBytesConfig를 사용해 8비트 양자화(PTQ) 적용
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quant_config, trust_remote_code=True, device_map="auto")
        # 기본 모델과 토크나이저 로드
        #base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, device_map="auto")

    else:
        # CPU: 일반 모델을 로드한 후 동적 양자화 적용 (특히 Linear 계층)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
        base_model.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # (필요 시 tokenizer.pad_token 설정)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 저장된 LoRA 어댑터 적용
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if torch.cuda.is_available() else None
            )
    # HuggingFacePipeline으로 래핑하여 LangChain의 LLM 인터페이스에 맞춥니다.
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm
