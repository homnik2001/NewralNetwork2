import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json


# Функция для загрузки модели и токенизатора
def load_model_and_tokenizer(model_dir):
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    return model, tokenizer


# Функция для выполнения предсказаний
def predict_words(model, tokenizer, words, threshold=0.9):
    # Токенизируем слова
    inputs = tokenizer(words, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Получаем предсказания (logits)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Преобразуем logits в вероятности (softmax)
    probabilities = torch.softmax(logits, dim=-1)

    # Получаем метки класса с наибольшей вероятностью
    predictions = torch.argmax(probabilities, dim=-1)

    # Фильтруем по классу "ethnonyms" (предполагаем, что он в индексе 0)
    predicted_labels = predictions.tolist()
    predicted_probs = probabilities.max(dim=-1).values.tolist()

    # Выводим слова, которые с вероятностью выше 90% относятся к классу "ethnonyms"
    ethnonyms_words = [
        words[i] for i in range(len(words))
        if predicted_labels[i] == 1 and predicted_probs[i] >= threshold  # Предполагаем, что "ethnonyms" имеет индекс 0
    ]

    return ethnonyms_words


# Главная функция
if __name__ == "__main__":
    # Путь к сохраненной модели и токенизатору
    model_dir = "/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/testStudy/bert/results/checkpoint-650"

    # Загружаем модель и токенизатор
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # Пример списка слов (вы можете загрузить его из файла, например)
    words_file_path = "/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/utilresult/screen_names.json"

    with open(words_file_path, 'r', encoding='utf-8') as f:
        words = json.load(f)  # Считываем список слов из файла

    # Получаем слова, которые с вероятностью выше 90% отнесены к классу "ethnonyms"
    ethnonyms_words = predict_words(model, tokenizer, words, threshold=0.9)

    # Выводим результат
    print("Слова, которые с 90% уверенностью относятся к классу 'ethnonyms':")
    print(ethnonyms_words)
