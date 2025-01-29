import pytest
from source.books_collector import BooksCollector


book1 = 'Властелин колец'
book1_2 = 'Гарри'
genre1 = 'Фэнтези'

book2 = 'Матрица'
genre2 = 'Научная фантастика'

book3 = 'Луна жестко стелет'
genre3 = 'Фантастика'

unknown_book = 'Неизвестная книга'
unknown_genre = 'Неизвестный жанр'

@pytest.fixture
def collector():
    return BooksCollector()

@pytest.mark.unit
def test_init():
    """Тест метода BooksCollector.__init__()"""

    # тест создания объекта с параметрами по умолчанию
    c = BooksCollector()
    assert c.available_genres == ['Фантастика', 'Фэнтези', 'Научная фантастика'], \
           "При создании BooksCollector() без параметров список доступных жанров должен быть дефолтным"
    assert c.genre_age_rating == ['Научная фантастика'], \
           "При создании BooksCollector() без параметров список возрастных жанров должен быть дефолтным"
    
    # тест создания объекта со специфичными параметрами
    c = BooksCollector(genres=[genre1, genre2], genre_age_rating=[])
    assert c.available_genres == [genre1, genre2] and c.genre_age_rating==[], \
           "При создании BooksCollector() со специфичными параметрами списоки доступных и возрастных жанров должны соответствовать переданным"

    # тест создания объекта с ошибочными параметрами
    with pytest.raises(AttributeError):
        BooksCollector(genres=[genre1, genre2], genre_age_rating=[unknown_genre])

@pytest.mark.unit
def test_add_new_book(collector):
    """Тест метода BooksCollector.add_new_book()"""

    # тест добавления книги
    collector.add_new_book(book1)
    assert collector.books_genre == {book1: None}

    # тест добавления книги повторно
    collector.add_new_book(book1)  # Дубликат
    assert collector.books_genre == {book1: None}
    
    # текст добавления второй книни
    collector.add_new_book(book2)
    assert collector.books_genre == {book1: None, book2: None}

@pytest.mark.unit
def test_set_book_genre_valid(collector):
    """Тест метода BooksCollector.set_book_genre_valid()"""
    
    # тест установки жанра книги
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    assert collector.get_book_genre(book1) == genre1

    # тест установки жанра несуществующей книги
    with pytest.raises(AttributeError):
        collector.add_new_book(book3)
        collector.set_book_genre(unknown_book, genre3)

    # тест установки несуществующего жанра
    with pytest.raises(AttributeError):
        collector.add_new_book(book2)
        collector.set_book_genre(book1, unknown_genre)

@pytest.mark.unit
def test_get_books_genre(collector):
    """Тест метода BooksCollector.get_books_genre()"""
    
    # тест получения словаря книг, когда еще ни одной книги не добавлено
    assert collector.get_books_genre() == {}

    # тест получения словаря книг, когда добавлена одна книга
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    assert collector.get_books_genre() == {book1: genre1}

    # тест получения словаря книг, когда добавлена вторая книга
    collector.add_new_book(book2)
    collector.set_book_genre(book2, genre2)
    assert collector.get_books_genre() == {book1: genre1, book2: genre2}


@pytest.mark.unit
def test_get_book_genre(collector):
    """Тест метода BooksCollector.get_book_genre()"""
    
    # тест получения жанра книги, когда жанр еще не определено
    collector.add_new_book(book1)
    assert collector.get_book_genre(book1) == None

    # тест получения жанра книги, когда жанр задан
    collector.set_book_genre(book1, genre1)
    assert collector.get_book_genre(book1) == genre1
    
    # тест получения жанра неизвестной книги
    with pytest.raises(AttributeError):
        collector.get_book_genre(unknown_book)

@pytest.mark.unit
def test_get_books_with_specific_genre(collector):
    """Тест метода BooksCollector.get_books_with_specific_genre()"""
    
    # зарегистрировать несколько книг с жанрами
    # две книги с жанром genre1
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    collector.add_new_book(book1_2)
    collector.set_book_genre(book1_2, genre1)
    # одна книга с жанром genre2
    collector.add_new_book(book2)
    collector.set_book_genre(book2, genre2)
    # одна книга без жанра
    collector.add_new_book(book3)  
    
    # тест получения списка книг, когда одна книга по жанру
    assert collector.get_books_with_specific_genre(genre2) == [book2]

    # тест получения списка книг, когда две книги по жанру
    assert sorted(collector.get_books_with_specific_genre(genre1)) == sorted([book1, book1_2])

    # тест получения списка книг, когда по жанру нет книг
    assert sorted(collector.get_books_with_specific_genre(genre3)) == []

    # тест получения книг по неизвестному жанру
    with pytest.raises(AttributeError):
        collector.get_books_with_specific_genre(unknown_genre)

@pytest.mark.unit
def test_get_books_for_children(collector):
    """Тест метода BooksCollector.get_books_for_children()"""
    
    # тест получения списка книг для детей, когда книг еще нет
    assert collector.get_books_for_children() == []
    
    # тест получения списка книг для детей, когда книги уже есть, но детских нет
    collector.add_new_book(book2)
    collector.set_book_genre(book2, genre2)
    assert collector.get_books_for_children() == []

    # тест получение детских книг, когда есть одна детская книга
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    assert collector.get_books_for_children() == [book1]

    # тест получение детских книг, когда есть две детские книга одного жанра
    collector.add_new_book(book1_2)
    collector.set_book_genre(book1_2, genre1)
    assert sorted(collector.get_books_for_children()) == sorted([book1, book1_2])

    # тест получение детских книг, когда есть три детские книга двух разных жанров
    collector.add_new_book(book3)
    collector.set_book_genre(book3, genre3)
    assert sorted(collector.get_books_for_children()) == sorted([book1, book1_2, book3])

@pytest.mark.unit
def test_add_book_in_favorites(collector):
    """Тест метода BooksCollector.add_book_in_favorites()"""

    # тест добавления книги в избранные
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    assert collector.favorites == []
    collector.add_book_in_favorites(book1)
    assert collector.favorites == [book1]

    # тест повтороного добавления книги в избранные
    collector.add_book_in_favorites(book1)
    assert collector.favorites == [book1]

    # тест, что новые книги не попадают автоматически в избранные
    collector.add_new_book(book2)
    collector.add_new_book(book3)
    assert collector.favorites == [book1]

    # тест добавление второй книги в избранные
    collector.add_book_in_favorites(book2)
    assert collector.favorites == [book1, book2]
    
    # тест добавления в избранное неизвестной книги
    with pytest.raises(AttributeError):
        collector.add_book_in_favorites(unknown_book)
    

@pytest.mark.unit
def test_delete_book_from_favorites(collector):
    """Тест метода BooksCollector.delete_book_from_favorites()"""
    # регистрация книг
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    collector.add_new_book(book2)
    collector.set_book_genre(book2, genre2)
    collector.add_new_book(book3)
    collector.set_book_genre(book3, genre3)

    # добавление книг в избранные
    collector.add_book_in_favorites(book1)
    collector.add_book_in_favorites(book2)
    collector.add_book_in_favorites(book3)

    # тест удаления книги из избранных неизвестной книги
    with pytest.raises(AttributeError):
        collector.delete_book_from_favorites(unknown_book)

    # контрольный тест того, что избранные созданы как ожидались
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book1, book2, book3])
    
    # тест удаления книги из избранных
    collector.delete_book_from_favorites(book1)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book2, book3])

    # тест удаления книги из избранных, когда её там уже/еще нет
    collector.delete_book_from_favorites(book1)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book2, book3])

    # тест удаления еще одной книги из избранных
    collector.delete_book_from_favorites(book2)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book3])

    # тест удаления еще последней книги из избранных
    collector.delete_book_from_favorites(book3)
    assert sorted(collector.get_list_of_favorites_books()) == []


@pytest.mark.unit
def test_get_list_of_favorites_books(collector):
    """Тест метода BooksCollector.get_list_of_favorites_books()"""
    
    # тест получения избранных при пустом списке книг
    assert sorted(collector.get_list_of_favorites_books()) == []

    # тест получения избранных когда книга уже есть, но еще не избранных
    collector.add_new_book(book1)
    collector.set_book_genre(book1, genre1)
    assert sorted(collector.get_list_of_favorites_books()) == []

    # тест получения избранных когда книга уже есть и она в избранных
    collector.add_book_in_favorites(book1)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book1])

    # тест получения избранных когда зарегистрировано несколько книг, но только одна в избранных
    collector.add_new_book(book2)
    collector.add_new_book(book3)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book1])

    # тест получения избранных когда зарегистрировано несколько книг и некоторые из них в избранных
    collector.add_book_in_favorites(book2)
    assert sorted(collector.get_list_of_favorites_books()) == sorted([book1, book2])
    