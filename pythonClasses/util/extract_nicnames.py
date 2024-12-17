import json


# Функция для чтения данных из файла и извлечения screen_name
def extract_screen_names_from_file(input_file_path, output_file_path):
    # Открываем файл и загружаем данные
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Извлечение screen_name из каждого объекта
    screen_names = [item['screen_name'] for item in data]

    # Запись результата в файл
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(screen_names, output_file, indent=4)

    print(f"Результат сохранен в файл: {output_file_path}")


# Пути к файлам
input_file_path = '/forKotlin/output.json'
output_file_path = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/utilresult/screen_names.json'

# Извлечение и сохранение результата
extract_screen_names_from_file(input_file_path, output_file_path)