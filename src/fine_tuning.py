from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import os

# 1. 사전 학습 모델 로드
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 2. 파인튜닝 데이터 준비
# 예시 데이터: 유사도 라벨(0.0 ~ 1.0)을 사용 (1.0: 매우 유사, 0.0: 전혀 유사하지 않음)
train_examples = [
    InputExample(texts=["이 문장은 긍정적 예시입니다.", "이 문장은 비슷한 의미를 갖습니다."], label=1.0),
    InputExample(texts=["이 문장은 긍정적 예시입니다.", "이 문장은 전혀 다른 의미입니다."], label=0.0),
    # 더 많은 예시 추가 (실제 파인튜닝에는 수천~수만 개의 예시가 필요합니다)
]

# DataLoader 설정 (배치 사이즈 조정)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. 손실 함수 설정: 여기서는 CosineSimilarityLoss 사용
train_loss = losses.CosineSimilarityLoss(model=model)

# (옵션) 4. 평가 데이터 설정 - 검증 데이터셋이 있다면 사용
# 예를 들어, 개발 데이터셋으로 평가하여 학습 중 모델 성능 모니터링
# evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_sentences1, dev_sentences2, dev_scores)

# 5. 모델 파인튜닝
num_epochs = 1  # 실제 학습 시 에포크 수를 늘려야 함
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 전체 학습의 10% 정도

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="output/finetuned-model",
    # evaluator=evaluator,  # 평가 데이터가 있을 경우 사용
    evaluation_steps=1000  # 평가 주기 설정(옵션)
)

# 파인튜닝 완료 후 모델 저장: output/finetuned-model 디렉토리에 저장됨

