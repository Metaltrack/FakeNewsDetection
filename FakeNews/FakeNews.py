# pip install transformers datasets torch scikit-learn pandas

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# -------------------------
# Load datasets
# -------------------------

fake_df = pd.read_csv(r"D:\Misc\college\internship\archive_news\News _dataset\Fake.csv")
true_df = pd.read_csv(r"D:\Misc\college\internship\archive_news\News _dataset\True.csv")

# Assign labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------
# Prepare text
# -------------------------

df["combined_text"] = (
    df["title"].fillna("") +
    " [SEP] " +
    df["text"].fillna("") +
    " [SEP] " +
    df["subject"].fillna("")
)

X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# -------------------------
# Tokenization
# -------------------------

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# -------------------------
# Load BERT model
# -------------------------

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# -------------------------
# Metrics
# -------------------------

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -------------------------
# Training setup
# -------------------------

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# -------------------------
# Train
# -------------------------

trainer.train()


# -------------------------
# Evaluate
# -------------------------

results = trainer.evaluate()

print(results)

news = """
NASA confirms discovery of a new Earth-like planet
Scientists report potential evidence...
"""

inputs = tokenizer(
    news,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1)

if prediction.item() == 1:
    print("True news")
else:
    print("Fake news")