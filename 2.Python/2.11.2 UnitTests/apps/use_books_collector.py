import sys
import os
if os.path.exists('source'):
    sys.path.insert(0, "source")
elif os.path.exists('../source'):
    sys.path.insert(0, "../source")

try:
    from books_collector import BooksCollector
except ImportError:
    print('Can not import BooksCollector')

# Пример использования класса
collector = BooksCollector()
# Добавление книг
collector.add_new_book('Властелин колец')
collector.add_new_book('Гарри Поттер')
collector.add_new_book('Матрица')
# Установка жанров
collector.set_book_genre('Властелин колец', 'Фантастика')
collector.set_book_genre('Гарри Поттер', 'Фэнтези')
collector.set_book_genre('Матрица', 'Научная фантастика')
# Получение жанров
print(collector.get_book_genre('Властелин колец')) # Фантастика
print(collector.get_book_genre('Гарри Поттер')) # Фэнтези
print(collector.get_book_genre('Матрица')) # Научная фантастика# Получение книг определенного жанра
fantasy_books = collector.get_books_with_specific_genre('Фантастика')
print(f'Книги в жанре Фантастики: {fantasy_books}')
# Получение всех книг жанра
all_books = collector.get_books_genre()
print('Все книги и их жанры:', all_books)
# Получение книг для детей
children_books = collector.get_books_for_children()
print('Книги для детей:', children_books)
# Добавление в избранное
collector.add_book_in_favorites('Властелин колец')
# Удаление из избранного
collector.delete_book_from_favorites('Гарри Поттер')
# Получение списка избранных
favorites_list = collector.get_list_of_favorites_books()
print('Список избранных книг:', favorites_list)
collector.delete_book_from_favorites('Властелин колец')
print('Список избранных книг после удаления книги из избранных:', favorites_list)
