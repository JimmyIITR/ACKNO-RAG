import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
import dataSetRead
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

MODEL_NAME = "bert-base-uncased"
LABEL_MAP = {"supported": 0, "refuted": 1, "cherrypicking": 2}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClaimDataset(Dataset):
    def __init__(self, claims, evidences, labels, tokenizer, max_length=128):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = f"Claim: {self.claims[idx]} Evidence: {self.evidences[idx]}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def initialize_model():
    model = BertForSequenceClassification.from_pretrained( MODEL_NAME, num_labels=len(LABEL_MAP))
    return model.to(DEVICE)

def setup_training(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}
    )

def classify_claim(model, tokenizer, claim, evidence):
    text = f"Claim: {claim} Evidence: {evidence}"
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).cpu().item()
    
    inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
    return inverse_label_map[pred]

def main():
    example_data = {
        "claim": ["Vaccines cause autism", "Coffee prevents cancer"],
        "evidence": [
            "Multiple studies disprove this claim",
            "One small study showed minor correlation"
        ],
        "label": ["refuted", "cherrypicking"]
    }
    # df = pd.DataFrame(example_data)
    # df.to_csv("claims_dataset.csv", index=False)

    train_df, val_df = dataSetRead.main()

    # Initialize components
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = initialize_model()

    # Create datasets
    train_dataset = ClaimDataset(
        train_df["claim"].tolist(),
        train_df["evidence"].tolist(),
        train_df["label"].tolist(),
        tokenizer
    )
    val_dataset = ClaimDataset(
        val_df["claim"].tolist(),
        val_df["evidence"].tolist(),
        val_df["label"].tolist(),
        tokenizer
    )

    # Train model
    trainer = setup_training(model, train_dataset, val_dataset)
    trainer.train()

    # Save model
    model.save_pretrained("./claim_classifier")
    tokenizer.save_pretrained("./claim_classifier")

    # Example inference
    test_claim = "Solar energy is inefficient"
    test_evidence = "Modern panels reach 22% efficiency but initial models were poor"
    prediction = classify_claim(model, tokenizer, test_claim, test_evidence)
    print(f"\nPrediction: {prediction}")

if __name__ == "__main__":
    main()