# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gensim.downloader as api
import numpy as np
import json
import os

from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Путь для сохранения и загрузки предобученной модели
pretrained_model_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/testStudy/word2vec/predtrain/word2vec-google-news-300'
trained_model_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/testStudy/word2vec/trained/word_classifier_model.pth'

# Проверяем наличие предобученной модели в указанной папке
def load_or_download_word2vec_model():
    if os.path.exists(pretrained_model_path):
        print(f"Загружаем модель из {pretrained_model_path}")
        word2vec_model = KeyedVectors.load(pretrained_model_path)
        print("Модель успешно загружена.")# Загружаем локально сохраненную модель
    else:
        print("Модель не найдена. Загружаем из интернета...")
        try:
            word2vec_model = api.load("word2vec-google-news-300")  # Скачиваем модель из gensim API
            print("Модель успешно загружена.")
            word2vec_model.save(pretrained_model_path)
            print("Модель успешно сохранена.")# Сохраняем модель локально для будущего использования
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return None
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
        if word in word2vec_model:
            embeddings.append(word2vec_model[word])
        else:
            # Если слово не найдено в модели, заменим его на вектор нулей (или случайный вектор)
            embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.array(embeddings)

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

# Обучение модели
# Обучение модели
# Обучение модели
def train_model(model, train_data, train_labels, class_weights, epochs=10, batch_size=64, learning_rate=0.0001):
    # Преобразуем данные в тензоры
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Определяем оптимизатор и критерий
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Обучение модели
    for epoch in range(epochs):
        model.train()
        total_loss = 0  # Для подсчета средней потери за эпоху
        num_batches = len(train_data) // batch_size  # Количество батчей
        for i in range(0, len(train_data), batch_size):
            # Мини-батч
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            optimizer.zero_grad()

            # Прямой проход
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Обратное распространение
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Суммируем потери для усреднения

            # Выводим информацию о текущем батче
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{i//batch_size + 1}/{num_batches}], Loss: {loss.item():.4f}')

        # После обработки всех батчей в эпохе, выводим среднюю потерю
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')



# Основная функция
def main():
    # Путь к файлам с данными
    file_paths = ['/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/ethnonyms.json',
                  '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/negative.json']

    # Загружаем или скачиваем модель Word2Vec
    word2vec_model = load_or_download_word2vec_model()

    # Загрузка и подготовка данных
    words, labels = load_data(file_paths)
    embeddings = get_word_embeddings(words, word2vec_model)

    # Разделяем на обучающую и тестовую выборки
    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Присваиваем веса классам (в зависимости от несбалансированности)
    class_weights = [1.0, 1.0]  # Здесь могут быть ваши веса, например: [0.1, 1.0] для негативного класса
    class_weights = np.array(class_weights)

    # Создаем модель
    input_size = word2vec_model.vector_size  # Размерность вектора слова
    hidden_size = 16  # Скрытый слой
    num_classes = 2  # Два класса
    model = WordClassifier(input_size, hidden_size, num_classes)

    # Обучаем модель
    train_model(model, train_data, train_labels, class_weights)

    # Сохраняем модель
    torch.save(model.state_dict(), trained_model_path)
    print(f"Модель сохранена в {trained_model_path}")


if __name__ == "__main__":
    main()
