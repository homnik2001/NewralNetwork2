import json


# Функция для чтения JSON-файла и извлечения "id"
import json

def extract_ids_from_json(file_name):
    try:
        # Чтение содержимого файла
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Извлечение значений "id"
        ids_set = {item["id"] for item in data if "id" in item}

        print("Извлечённые ID:")
        for value in ids_set:
            print(value)

        return ids_set

    except FileNotFoundError:
        print("Файл не найден.")
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



# Пример использования
file_name = "/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/output.json"  # Замените на путь к вашему файлу
ids_dict = extract_ids_from_json(file_name)


input_file = "/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/users.json"
output_file = "/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/filtred_users.json"
# Загрузить словарь ids_dict



with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
filtred = list()
for obj in data:
    if obj['id'] in ids_dict:
        filtred.append(obj)

# Отфильтровать объекты

# Сохранить в новый файл
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtred, f, ensure_ascii=False, indent=4)

print(f"Фильтрация завершена. Найдено объектов: {len(filtred)}")
