import json

# Открытие исходного JSON файла для чтения
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/filtred_users.json', 'r', encoding='utf-8') as infile:
    # Считываем все данные из файла
    data = json.load(infile)

# Новый список для хранения отфильтрованных данных
filtered_data = []

# Фильтрация данных: оставляем только нужные поля
for item in data:
    filtered_item = {
        'id': item.get('id'),
        'bdate': item.get('bdate'),
        'schools': item.get('schools'),
        'sex': item.get('sex'),
        'universities': item.get('universities')
    }
    filtered_data.append(filtered_item)

# Запись отфильтрованных данных в новый JSON файл
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/forKotlin/finalFiltred_users.json', 'w', encoding='utf-8') as outfile:
    json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

print("Фильтрация завершена, результат записан в 'output.json'.")
