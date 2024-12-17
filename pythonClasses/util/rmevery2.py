def filter_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        line_number = 0
        for line in infile:
            line_number += 1
            print(line_number)
            if line_number % 10 == 0:  # Пропускаем каждую вторую строку (с нумерацией с 1)
                outfile.write(line)

# Путь к исходному файлу и файлу для записи результата
input_file = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/negativeDiv100.json'
output_file = '/Users/nikitakhomenko/PycharmProjects/NeuralNetworkLab2/pythonProject/classesdata/negative.json'

# Вызов функции для фильтрации строк
filter_lines(input_file, output_file)

print(f"Удаление каждой второй строки завершено. Результат записан в {output_file}")