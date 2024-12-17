import json

# Чтение исходного JSON файла
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/words.json', 'r') as file:
    data = json.load(file)

# Извлечение всех ключей и создание списка
keys_list = list(data.keys())

# Запись преобразованного результата в новый файл
with open('/classesdata/savenegative.json', 'w') as file:
    json.dump(keys_list, file, indent=4)

# Для вывода в консоль:
#print(keys_list)
