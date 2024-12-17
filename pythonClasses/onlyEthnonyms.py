import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset


# Custom Dataset for tokenized text
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.labels[idx]
        }


# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function to get word embeddings
def get_word_embeddings(words, tokenizer, model):
    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  # Запрос скрытых состояний
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1].mean(dim=1)  # Среднее по последнему слою
    return embeddings


# Function to find similar words with similarity threshold
def find_similar_words(target_word, word_list, tokenizer, model, similarity_threshold=0.9):
    target_embedding = get_word_embeddings([target_word], tokenizer, model)
    embeddings = get_word_embeddings(word_list, tokenizer, model)

    # Нормализуем эмбеддинги
    target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    similarities = cosine_similarity(target_embedding.cpu().numpy(), embeddings.cpu().numpy())

    # Получаем индексы слов с схожестью выше порога
    similar_indices = np.where(similarities[0] > similarity_threshold)[0]

    # Возвращаем слова с высоким сходством
    similar_words = [word_list[i] for i in similar_indices]

    return similar_words


# Fine-tuning the model (можно оставить как есть или настроить)
def fine_tune_model(train_words, tokenizer, model, output_dir='./fine_tuned_model'):
    inputs = tokenizer(train_words, return_tensors='pt', padding=True, truncation=True)
    dataset = TextDataset(inputs['input_ids'], inputs['input_ids'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Main function
if __name__ == "__main__":
    # Пути к файлам
    train_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/ethnonyms.json'  # Путь к обучающим данным
    test_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/utilresult/screen_names.json'  # Путь к тестовым данным
    output_model_dir = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/models'  # Путь для сохранения обученной модели

    # Загрузка обучающих и тестовых слов
    train_words = load_json(train_file_path)
    test_words = load_json(test_file_path)

    # Загрузка предобученного токенизатора и модели
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Устанавливаем pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # Тонкая настройка модели (если требуется)
    fine_tune_model(train_words, tokenizer, model, output_dir=output_model_dir)

    # Загружаем дообученную модель
    tokenizer = GPT2Tokenizer.from_pretrained(output_model_dir)
    model = GPT2LMHeadModel.from_pretrained(output_model_dir)

    # Оценка модели по тестовым словам
    for word in test_words:
        similar_words = find_similar_words(word, train_words, tokenizer, model, similarity_threshold=0.9)
        if similar_words:  # Если есть похожие слова
            print(f"Words similar to '{word}' with similarity > 90%: {', '.join(similar_words)}")
