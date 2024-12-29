import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the cleaned data
data = pd.read_csv('./data/cleaned_data.csv')

# Drop null values and ensure all entries are strings
data = data.dropna(subset=['cleaned_resume'])
data['cleaned_resume'] = data['cleaned_resume'].astype(str)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokens = tokenizer(list(data['cleaned_resume']), truncation=True, padding=True, max_length=512, return_tensors="pt")

print("Tokenization done")

# Create Hugging Face Dataset
dataset = Dataset.from_dict({
    "input_ids": tokens['input_ids'],
    "attention_mask": tokens['attention_mask'],
    "labels": data['encoded_category'].tolist()
})

# Split data into train-test sets
dataset = dataset.train_test_split(test_size=0.2)

print("Data split prepared")

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(data['encoded_category'].unique()))

print("Model loaded Succesfully")

# Training arguments with checkpoint saving
output_dir = "./logs"
training_args = TrainingArguments(
    output_dir=output_dir,                     # Directory to save checkpoints and logs
    eval_strategy="epoch",                     # Evaluate at the end of each epoch
    save_strategy="epoch",                     # Save a checkpoint at the end of each epoch
    save_total_limit=20,                       # Keep only the last 20 checkpoints
    learning_rate=2e-5,                        # Learning rate
    per_device_train_batch_size=8,             # Batch size for training
    per_device_eval_batch_size=8,              # Batch size for evaluation
    num_train_epochs=20,                       # Start with 20 epochs
    weight_decay=0.01,                         # Weight decay
    logging_dir=output_dir,                    # Directory for logs
    logging_steps=100,                         # Log after every 100 steps
    load_best_model_at_end=True,               # Load the best model after training
)

# Check for existing checkpoints
checkpoint_path = None
if os.path.isdir(output_dir) and any("checkpoint" in d for d in os.listdir(output_dir)):
    checkpoint_path = output_dir  # Resume training from checkpoint

from sklearn.metrics import accuracy_score

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model with checkpoint resumption
print("Starting training...")
if checkpoint_path:
    print(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("No checkpoint found. Starting from scratch.")
    trainer.train()

# Save the fine-tuned model and tokenizer
final_output_dir = "./models/resume_classifier_model"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"Training complete. Model saved to '{final_output_dir}'.")

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)
