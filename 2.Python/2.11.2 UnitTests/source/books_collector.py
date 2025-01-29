class BooksCollector:
    def __init__(self, genres=['Фантастика', 'Фэнтези', 'Научная фантастика'], 
                 genre_age_rating = ['Научная фантастика']):
        # Словарь: НазваниеКниги:ЖанрКниги
        self.books_genre = {}
        # избранные книги
        self.favorites = []
        self.available_genres = genres
        # проверить, что все "возрастные жанры есть в доступных"
        unknown_genres = [g for g in genre_age_rating if g not in self.available_genres]
        if len(unknown_genres) > 0:
            raise AttributeError(f"Неизвестные возрастные жанры: {unknown_genres}")
        
        self.genre_age_rating = genre_age_rating

    def add_new_book(self, name):
        """Добавить новую книгку в словарь"""
        if name not in self.books_genre:
            self.books_genre[name] = None

    def set_book_genre(self, name, genre):
        """Установить жанр для книги"""
        if genre in self.available_genres:
            if name in self.books_genre:
                self.books_genre[name] = genre
            else:
                raise AttributeError(f"Книга '{name}' не зарегистрирована")
        else:
            raise AttributeError(f"Жанр '{genre}' отсутствует в списке доступных жанров")

    def get_books_genre(self):
        """Получить все книги"""
        return self.books_genre

    def get_book_genre(self, name):
        """Получить жанр для книги"""
        if name in self.books_genre:
            return self.books_genre.get(name)
        else:
            raise AttributeError(f"Книга '{name}' не зарегистрирована")

    def get_books_with_specific_genre(self, genre):
        """Получить список книг с заданным жанром"""
        if genre in self.available_genres:
            return [book for book, g in self.books_genre.items() if g == genre]
        else:
            raise AttributeError(f"Жанр '{genre}' отсутствует в списке доступных жанров")

    def get_books_for_children(self):
        """Получить список книг для детей"""
        return [book for book, genre in self.books_genre.items() if genre not in self.genre_age_rating]

    def add_book_in_favorites(self, name):
        """Добавить книгу в избранное"""
        if name in self.books_genre:
            if name not in self.favorites:
                self.favorites.append(name)
        else:
            raise AttributeError(f"Книга '{name}' не зарегистрирована")

    def delete_book_from_favorites(self, name):
        """Удалить книгу из избранного"""
        if name in self.books_genre:
            if name in self.favorites:
                self.favorites.remove(name)
        else:
            raise AttributeError(f"Книга '{name}' не зарегистрирована")

    def get_list_of_favorites_books(self):
        """Получить список избранных книг"""
        return self.favorites
    
