"""
Задание по функциональному программированию.

igel2000@gmail.com
"""
__author__ = 'igel2000'

from typing import List, Iterable, Dict
import itertools
import functools

# region Задача 1 Фильтрация данных
def three_or_more(s: str) -> bool:
    """Проверить, что переданная строка длинной 3 и более символа
    Return
    -------
    * True, если строка длинной 3 и более символов 
    * False, если None или строку менее 3 символов.

    Parameters
    ----------
    s: str - проверяемая строка
    
    Raises
    ------
    TypeError - если передать не строку
    """
    if s is None:
        return False
    if not isinstance(s, str):
        raise TypeError('Параметр s должен быть строкой')
    result = True if len(s) >= 3 else False
    return result

def filter_more_than_three(l: Iterable) -> List:
    """Выбрать из l строки длинной более трех символов и вернуть их в виде списка
    
    Если передать None или неитерируемый объект - вернуть пустой список
    """
    if l is None or not isinstance(l, Iterable):
        return []
    return [i for i in l if three_or_more(i)] # альтернатива - list(filter(three_or_more, l))

def println(l: Iterable) -> None:
    """ Вывести список на экран - каждый элемент в отдельной строке"""
    for i in l:
        print(i)

#endregion

#region Задача 2 Вложенные функции
def is_int(s: str) -> bool:
    """Проверить, что строка содержит целое число"""
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def get_list(int_as_str: str) -> List:
    """Преобразовать строку с числами в список чисел"""
    if int_as_str is None or not isinstance(int_as_str, str) or len(int_as_str) == 0:
        return []
    return list(map(int, filter(lambda i: is_int(i), int_as_str.split(" "))))

def sort_func(f, *args, **kwargs) -> List:
    """Отсортировать результать функции f, передав в неё все остальные параметры"""
    result_f = f(*args, **kwargs)
    sorted_result = sorted(result_f)
    return sorted_result

#endregion        

#region Задача 3 Перестановки строк
def get_permutations(s: str, n: int) -> List:
    """Получить все перестановки длинной n из строки s в лексикографическом порядке"""
    s1 = sorted(s) # отсортировать, чтобы перестановки были в лексикографическом порядке
    r = ["".join(p) for p in itertools.permutations(s1, n)]
    return r
#endregion

#region Задача 4 Комбинации символов
def get_combinations(s: str, k: int) -> List:
    """Получить все комбинации длинной не более k из строки s"""
    r = ["".join(p) for i in range(1, k+1) for p in itertools.combinations(s, i)]
    return r
#endregion

#region Задача 5 Функция с частичными аргументами
def sort_users_by_age(users: List, ascending = True) -> Dict:
    """Сортировать список users по возрасту"""
    if users is None or len(users)==0 or not isinstance(users, List):
        return []
    return sorted(users, key = lambda x: x['age'], reverse = not ascending)

sort_users_by_age_ascending = functools.partial(sort_users_by_age, ascending = True)
sort_users_by_age_ascending.__doc__ = "Сортировать список users по возрастанию возраста" 


sort_users_by_age_descending = functools.partial(sort_users_by_age, ascending = False)
sort_users_by_age_descending.__doc__ = "Сортировать список users по убыванию возраста" 

# endregion
