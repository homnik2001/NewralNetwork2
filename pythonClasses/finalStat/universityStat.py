import json

# Чтение первого JSON файла (с никнеймами и типами)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/classToNickName.json', 'r', encoding='utf-8') as file:
    nicknames_to_type = json.load(file)

# Чтение второго JSON файла (с пользователями)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/final/universities_valid.json', 'r', encoding='utf-8') as file:
    users_data = json.load(file)

# Структура для подсчета статистики
type_stats = {}
total_users = 0

# Проходим по каждому пользователю во втором файле
for user in users_data:
    user_id = user.get('id')

    # Проверяем, есть ли такой user_id в первом файле (сопоставление с никнеймами)
    if user_id in nicknames_to_type:
        user_type = nicknames_to_type[user_id]

        # Увеличиваем общее количество пользователей
        total_users += 1

        # Увеличиваем счетчик для этого типа
        if user_type not in type_stats:
            type_stats[user_type] = 0
        type_stats[user_type] += 1

# Выводим процентное распределение по типам
print("Процентное распределение среди студентов:")

for user_type, count in type_stats.items():
    # Вычисляем процентное распределение
    percentage = (count / total_users) * 100 if total_users > 0 else 0
    print(f"{user_type}: {percentage:.2f}%")

# Если не было найдено пользователей, то выводим соответствующее сообщение
if total_users == 0:
    print("Не было найдено пользователей с соответствующими никнеймами.")
