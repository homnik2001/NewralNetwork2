import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

# Нейронная сеть для классификации
class WordClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(WordClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Проверяем наличие предобученной модели Word2Vec
def load_word2vec_model(pretrained_model_path):
    if os.path.exists(pretrained_model_path):
        print(f"Загружаем Word2Vec модель из {pretrained_model_path}")
        word2vec_model = KeyedVectors.load(pretrained_model_path)
    else:
        raise FileNotFoundError(f"Модель Word2Vec не найдена по пути {pretrained_model_path}.")
    return word2vec_model

# Загружаем данные из файлов
def load_data(file_paths):
    data = []
    labels = []
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as file:
            words = json.load(file)
        data.extend(words)
        labels.extend([idx] * len(words))
    return data, labels

# Преобразуем слова в векторные представления
def get_word_embeddings(words, word2vec_model):
    embeddings = []
    for word in words:
        word = word.lower()  # Приведение к нижнему регистру
        if word in word2vec_model:
            embeddings.append(word2vec_model[word])
        else:
            # Если слово не в модели, можно использовать случайный вектор или средний вектор
            embeddings.append(np.random.normal(0, 1, word2vec_model.vector_size))  # случайный вектор
            # или embeddings.append(np.zeros(word2vec_model.vector_size))  # вектор из нулей
    return np.array(embeddings)

# Обучение модели
def train_model(model, train_data, train_labels, class_weights, epochs=10, batch_size=32, learning_rate=0.001):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Класс для предсказания
class WordPredictor:
    def __init__(self, model_path, word2vec_model_path, class_data_paths):
        self.model = None
        self.word2vec_model = None
        self.class_words = []
        self.class_labels = []

        # Загрузка модели Word2Vec
        self.word2vec_model = load_word2vec_model(word2vec_model_path)

        # Загрузка слов классов
        self.load_class_words(class_data_paths)

        # Загрузка обученной модели
        self.load_trained_model(model_path)

    def load_class_words(self, class_data_paths):
        for idx, path in enumerate(class_data_paths):
            with open(path, 'r', encoding='utf-8') as file:
                words = json.load(file)
                self.class_words.append(words)
                self.class_labels.append(os.path.basename(path).split('.')[0])
        print(f"Слова для классов успешно загружены: {self.class_labels}")

    def load_trained_model(self, model_path):
        input_size = self.word2vec_model.vector_size
        hidden_size = 16
        num_classes = len(self.class_labels)

        self.model = WordClassifier(input_size, hidden_size, num_classes)
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Модель успешно загружена из {model_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Ошибка при загрузке модели. Проверьте количество классов: {e}")

    def predict(self, words):
        results = {}
        embeddings = get_word_embeddings(words, self.word2vec_model)  # преобразуем слова в векторы

        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(embeddings)
            predicted_classes = torch.argmax(outputs, dim=1).numpy()

        for i, word in enumerate(words):
            predicted_class = predicted_classes[i]
            class_name = self.class_labels[predicted_class]
            class_words = self.class_words[predicted_class]

            closest_word = None
            min_distance = float('inf')
            word_vector = embeddings[i].numpy()

            for class_word in class_words:
                if class_word in self.word2vec_model:
                    class_word_vector = self.word2vec_model[class_word]
                    distance = cosine(word_vector, class_word_vector)
                    if distance < min_distance:
                        min_distance = distance
                        closest_word = class_word

            results[word] = {
                'predicted_class': class_name,
                'closest_word': closest_word,
                'distance': min_distance
            }

        return results

# Основная функция
if __name__ == "__main__":
    # Пути к данным
    model_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/testStudy/word2vec/trained/word_classifier_model.pth'
    word2vec_model_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/testStudy/word2vec/predtrain/word2vec-google-news-300'
    class_data_paths = [
        '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/ethnonyms.json',
        '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/negative.json'
    ]
    test_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/utilresult/screen_names.json'

    # Загрузка тестовых слов
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_words = json.load(file)

    # Создаем предсказатель
    predictor = WordPredictor(model_path, word2vec_model_path, class_data_paths)

    # Делаем предсказания
    predictions = predictor.predict(test_words)

    # Выводим результаты
    for word, result in predictions.items():
        print(f"Слово: {word}")
        print(f"  Предсказанный класс: {result['predicted_class']}")
        print(f"  Ближайшее слово: {result['closest_word']}")
        print(f"  Косинусное расстояние: {result['distance']:.4f}")
