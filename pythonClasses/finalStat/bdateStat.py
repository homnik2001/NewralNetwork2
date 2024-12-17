import json
from datetime import datetime


# Функция для вычисления возраста на основе bdate
def calculate_age(bdate):
    try:
        bdate_obj = datetime.strptime(bdate, "%d.%m.%Y")
        today = datetime.today()
        age = today.year - bdate_obj.year - ((today.month, today.day) < (bdate_obj.month, bdate_obj.day))
        return age
    except ValueError:
        return None  # Если дата невалидна, возвращаем None


# Чтение первого JSON файла (с никнеймами и типами)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/classToNickName.json', 'r', encoding='utf-8') as file:
    nicknames_to_type = json.load(file)

# Чтение второго JSON файла (с пользователями)
with open('/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/finalStat/final/bdate_valid.json', 'r', encoding='utf-8') as file:
    users_data = json.load(file)

# Создадим структуру для подсчета статистики
type_stats = {}
total_users = 0
total_older_25 = 0
total_younger_25 = 0

# Проходим по каждому элементу второго файла (пользователю)
for user in users_data:
    user_id = user.get('id')
    bdate = user.get('bdate')

    # Проверяем, есть ли такой user_id в первом файле (сопоставление с никнеймами)
    if user_id in nicknames_to_type:
        # Вычисляем возраст пользователя
        age = calculate_age(bdate)

        # Если возраст не определен, пропускаем этого пользователя
        if age is not None:
            user_type = nicknames_to_type[user_id]

            # Увеличиваем общее количество пользователей
            total_users += 1

            # Увеличиваем количество по возрастным категориям
            if age >= 25:
                total_older_25 += 1
            else:
                total_younger_25 += 1

            # Увеличиваем статистику для каждого типа
            if user_type not in type_stats:
                type_stats[user_type] = {'older': 0, 'younger': 0, 'total': 0}

            type_stats[user_type]['total'] += 1

            if age >= 25:
                type_stats[user_type]['older'] += 1
            else:
                type_stats[user_type]['younger'] += 1

# Выводим статистику: процентное распределение по типам и возрастам

print("Распределение по типам в процентах:")

for user_type, stats in type_stats.items():
    total = stats['total']
    older = stats['older']
    younger = stats['younger']

    # Вычисляем процентное распределение для каждого типа по возрастам
    older_percentage = (older / total_users) * 100 if total_users > 0 else 0
    younger_percentage = (younger / total_users) * 100 if total_users > 0 else 0

    # Процент по типам в общем контексте
    type_percentage = (total / total_users) * 100 if total_users > 0 else 0

    print(f"\nТип: {user_type}")
    print(f"  Всего пользователей этого типа: {total} ({type_percentage:.2f}%)")
    print(f"  Моложе 25 лет: {younger_percentage:.2f}%")
    print(f"  Старше или равно 25 лет: {older_percentage:.2f}%")

# Выводим общее распределение по возрастам (моложе/старше 25 лет)
if total_users > 0:
    total_older_25_percentage = (total_older_25 / total_users) * 100
    total_younger_25_percentage = (total_younger_25 / total_users) * 100

    print("\nОбщее распределение по возрастам:")
    print(f"  Моложе 25 лет: {total_younger_25_percentage:.2f}%")
    print(f"  Старше или равно 25 лет: {total_older_25_percentage:.2f}%")
    print(f"  Всего проанализированных пользователей: {total_users}")
