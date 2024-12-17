import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, AddedToken
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from JSON files
data_dir = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata'
files = ['negative.json', 'ethnonyms.json']
words = []
labels = []

for i, file_name in enumerate(files):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        words.extend(data)
        labels.extend([i] * len(data))  # 0 for negative, 1 for ethnonyms

# Compute class weights
unique_labels = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
# Adjust weight for negative class to be 10 times smaller
class_weights[0] = 1
class_weights[1] = 1
class_weights = torch.tensor(class_weights, dtype=torch.float)
# Tokenize the data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

custom_tokens = ["<ethnonym>", "<custom_word>"]

# Добавляем кастомные токены в словарь токенизатора с параметром single_word=True
added_tokens = [AddedToken(token, single_word=True) for token in custom_tokens]
tokenizer.add_tokens(added_tokens)
# Добавляем их в токенизатор
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=50)

# Create Dataset
dataset_dict = {
    'text': words,
    'labels': labels
}
dataset = Dataset.from_dict(dataset_dict)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert to pandas DataFrame
df = tokenized_dataset.to_pandas()

# Perform stratified splitting using scikit-learn
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['labels'])

# Convert back to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Set up the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Define loss function with adjusted class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(axis=-1) == p.label_ids).mean()},
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/modelss/RusskiBert')
tokenizer.save_pretrained('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/modelss/RusskiBert')