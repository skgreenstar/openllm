import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset

# Dummy Dataset: 데이터 파일이 없을 경우 사용할 기본 예시
class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = [
            "안녕하세요, 오늘 날씨가 참 좋네요.",
            "LoRA를 활용한 파인튜닝 예제입니다.",
            "이 모델은 도메인 특화 태스크에 최적화됩니다."
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        # 배치 차원 제거 및 레이블 설정
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

# DynamicDataset: 파일에서 읽은 텍스트 데이터를 이용하는 데이터셋
class DynamicDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

# 데이터 파일 경로가 주어지면 파일에서 데이터를 읽고, 없으면 DummyDataset 사용
def load_dynamic_dataset(tokenizer, data_file=None):
    if data_file is None:
        print("학습 데이터 파일이 제공되지 않았습니다. DummyDataset을 사용합니다.")
        return DummyDataset(tokenizer)
    else:
        texts = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        if not texts:
            print("파일에서 데이터를 읽지 못했습니다. DummyDataset을 사용합니다.")
            return DummyDataset(tokenizer)
        return DynamicDataset(tokenizer, texts)

def main():
    parser = argparse.ArgumentParser(description="LoRA 기반 파인튜닝 스크립트")
    parser.add_argument("--data_file", type=str, default=None, help="학습 데이터 파일 경로 (각 줄에 하나의 샘플)")
    args = parser.parse_args()

    # 1. 사전학습 모델과 토크나이저 로드 (Mistral-7B 예시)
    model_name = "mistralai/Mistral-7B-v0.3"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # pad token 설정: pad token이 없으면 eos token을 pad token으로 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 또는 새로운 pad token 추가:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    # 2. LoRA 설정 및 적용
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 인과관계 언어 모델에 적용
        inference_mode=False,          # 학습 모드
        r=8,                           # Low-Rank 행렬 차원
        lora_alpha=32,                 # 스케일링 팩터
        lora_dropout=0.1,               # 드롭아웃 비율
        target_modules=["q_proj", "v_proj"]  # 예시: 쿼리와 값 프로젝션 레이어에 적용 (모델에 따라 조정)
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 적용 후, 모델 파라미터 수:", sum(p.numel() for p in model.parameters()))

    # 3. 동적 학습 데이터셋 로드
    train_dataset = load_dynamic_dataset(tokenizer, args.data_file)

    # 4. Data Collator 설정 (MLM이 아닌 CAUSAL LM 학습)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. TrainingArguments 및 Trainer 설정
    training_args = TrainingArguments(
        output_dir="output/finetuned_lora",
        per_device_train_batch_size=1,
        num_train_epochs=1,            # 실제 학습 시 에포크 수를 늘리세요.
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        learning_rate=2e-4,
        fp16=True                      # GPU 환경에서 fp16 사용 시 속도 개선
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 6. 모델 파인튜닝 실행
    trainer.train()

    # 7. 파인튜닝된 모델 저장
    model.save_pretrained("output/finetuned_lora")

if __name__ == "__main__":
    main()

