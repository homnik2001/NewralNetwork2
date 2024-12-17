import json

# Чтение первого файла (с id и screen_name)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/output.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

# Чтение второго файла (с id, bdate, schools, sex и т.д.)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/finalFiltred_users.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Создаем словарь для быстрого поиска screen_name по id из первого файла
id_to_screenname = {item['id']: item['screen_name'] for item in data1}

# Обрабатываем второй файл: если id совпадает, заменяем его на screen_name
for item in data2:
    if item['id'] in id_to_screenname:
        item['id'] = id_to_screenname[item['id']]  # заменяем id на screen_name

# Записываем результат в третий файл
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/final.json', 'w', encoding='utf-8') as output_file:
    json.dump(data2, output_file, ensure_ascii=False, indent=4)

print("Фильтрация завершена, результат записан в 'output.json'.")
