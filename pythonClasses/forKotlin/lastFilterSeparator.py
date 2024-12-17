import json
import re

# Функция для проверки формата bdate
def is_valid_bdate(bdate):
    # Проверяем, что bdate является строкой и соответствует формату dd.M.YYYY
    if isinstance(bdate, str):
        return bool(re.match(r'^\d{2}\.\d{1,2}\.\d{4}$', bdate))
    return False

# Чтение исходного файла
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/final.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# Списки для хранения отфильтрованных данных
bdate_valid = []
schools_valid = []
universities_valid = []

# Обрабатываем данные и фильтруем по условиям
for item in data:
    # Условие 1: Проверка формата bdate
    if is_valid_bdate(item.get('bdate', '')):
        bdate_valid.append(item)

    # Условие 2: schools не null и не пустой
    if item.get('schools') and isinstance(item['schools'], list) and len(item['schools']) > 0:
        schools_valid.append(item)

    # Условие 3: universities не null и не пустой
    if item.get('universities') and isinstance(item['universities'], list) and len(item['universities']) > 0:
        universities_valid.append(item)

# Запись отфильтрованных данных в разные файлы
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/final/bdate_valid.json', 'w', encoding='utf-8') as outfile_bdate:
    json.dump(bdate_valid, outfile_bdate, ensure_ascii=False, indent=4)

with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/final/schools_valid.json', 'w', encoding='utf-8') as outfile_schools:
    json.dump(schools_valid, outfile_schools, ensure_ascii=False, indent=4)

with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/final/universities_valid.json', 'w', encoding='utf-8') as outfile_universities:
    json.dump(universities_valid, outfile_universities, ensure_ascii=False, indent=4)

print("Фильтрация завершена. Результаты записаны в файлы:")
print("- bdate_valid.json")
print("- schools_valid.json")
print("- universities_valid.json")
