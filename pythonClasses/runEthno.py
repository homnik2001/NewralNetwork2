import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel


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


# Function to find similar words
def find_similar_words(target_word, word_list, tokenizer, model, top_k=3):
    target_embedding = get_word_embeddings([target_word], tokenizer, model)
    embeddings = get_word_embeddings(word_list, tokenizer, model)
    similarities = cosine_similarity(target_embedding.cpu().numpy(), embeddings.cpu().numpy())
    similar_indices = np.argsort(similarities[0])[::-1][:top_k]
    return [word_list[i] for i in similar_indices]


# Main function
if __name__ == "__main__":
    # Пути к файлам
    trained_model_dir = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/models'
    test_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/utilresult/screen_names.json'
    train_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/ethnonyms.json'

    # Загрузка обученной модели и токенайзера
    tokenizer = GPT2Tokenizer.from_pretrained(trained_model_dir)
    model = GPT2LMHeadModel.from_pretrained(trained_model_dir)

    # Устанавливаем pad_token (на случай, если модель требует его)
    tokenizer.pad_token = tokenizer.eos_token

    # Загрузка данных
    train_words = load_json(train_file_path)  # Слова для сравнения
    test_words = load_json(test_file_path)  # Слова для тестирования

    # Поиск похожих слов
    for word in test_words:
        similar_words = find_similar_words(word, train_words, tokenizer, model, top_k=3)
        print(f"Words similar to '{word}': {similar_words}")
