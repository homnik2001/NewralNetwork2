import json

# Чтение первого JSON файла (с никнеймами и типами)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/classToNickName.json', 'r', encoding='utf-8') as file:
    nicknames_to_type = json.load(file)

# Чтение второго JSON файла (с пользователями)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/final/final.json', 'r', encoding='utf-8') as file:
    users_data = json.load(file)

# Создадим структуру для подсчета статистики
type_stats = {}
total_users = 0
total_male = 0
total_female = 0

# Проходим по каждому элементу второго файла (пользователю)
for user in users_data:
    user_id = user.get('id')
    sex = user.get('sex')

    # Проверяем, есть ли такой user_id в первом файле (сопоставление с никнеймами)
    if user_id in nicknames_to_type:
        # Если поле sex не определено, пропускаем этого пользователя
        if sex is not None:
            user_type = nicknames_to_type[user_id]

            # Увеличиваем общее количество пользователей
            total_users += 1

            # Увеличиваем количество по полу
            if sex == 1:  # Мужской пол
                total_male += 1
            elif sex == 2:  # Женский пол
                total_female += 1

            # Увеличиваем статистику для каждого типа
            if user_type not in type_stats:
                type_stats[user_type] = {'male': 0, 'female': 0, 'total': 0}

            type_stats[user_type]['total'] += 1

            if sex == 1:
                type_stats[user_type]['male'] += 1
            elif sex == 2:
                type_stats[user_type]['female'] += 1

# Выводим статистику: процентное распределение по типам и полам

print("Распределение по типам и полу в процентах:")

for user_type, stats in type_stats.items():
    total = stats['total']
    male = stats['male']
    female = stats['female']

    # Вычисляем процентное распределение для каждого типа по полу
    male_percentage = (male / total_users) * 100 if total_users > 0 else 0
    female_percentage = (female / total_users) * 100 if total_users > 0 else 0

    # Процент по типам в общем контексте
    type_percentage = (total / total_users) * 100 if total_users > 0 else 0

    print(f"\nТип: {user_type}")
    print(f"  Всего пользователей этого типа: {total} ({type_percentage:.2f}%)")
    print(f"  Мужчины: {male_percentage:.2f}%")
    print(f"  Женщины: {female_percentage:.2f}%")

# Выводим общее распределение по полу
if total_users > 0:
    total_male_percentage = (total_male / total_users) * 100
    total_female_percentage = (total_female / total_users) * 100

    print("\nОбщее распределение по полу:")
    print(f"  Мужчины: {total_male_percentage:.2f}%")
    print(f"  Женщины: {total_female_percentage:.2f}%")
    print(f"  Всего проанализированных пользователей: {total_users}")
