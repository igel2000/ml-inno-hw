import pytest
import func

#region Задача 1
@pytest.mark.parametrize("s, expected", [(None, False),
                                         ("", False),
                                         ("s", False),
                                         ("ss", False),
                                         ("sss", True),
                                         ("ssss", True)
                                         ])
def test_three_or_more(s, expected):
    assert func.three_or_more(s) == expected

@pytest.mark.parametrize("l, expected", [([], []),
                                         (["s"], []),
                                         (("s", "sss"), ["sss"]),
                                         (["s", "sss", "dddd"], ["sss", "dddd"])
                                         ])
def filter_more_than_three(l, expected):
    assert func.filter_more_than_three(l) == expected
#endregion

#region Задача 2
@pytest.mark.parametrize("l, expected", [(None, []),
                                         ("", []),
                                         ("1", list([1])),
                                         ("1 2", [1, 2]),
                                         ("1 2 3", [1, 2, 3])
                                         ])
def test_get_list(l, expected):
    assert func.get_list(l) == expected
    
@pytest.mark.parametrize("l, expected", [("1 2 3", [1, 2, 3]),
                                         ("3 2 1", [1, 2, 3])
                                         ])
def test_sort_func(l, expected):
    assert func.sort_func(func.get_list, l) == expected
#endregion


#region Задача 3
def test_get_permutations():
    r = func.get_permutations("bca", 2)
    assert r == ['ab', 'ac', 'ba', 'bc', 'ca', 'cb']
#endregion

#region Задача 4
def test_get_combinations():
    r = func.get_combinations("dbcad", 3)
    assert r == ['d', 'b', 'c', 'a', 'd', 'db', 'dc', 'da', 'dd', 'bc', 'ba', 'bd', 'ca', 'cd', 'ad', 'dbc', 'dba', 'dbd', 'dca', 'dcd', 'dad', 'bca', 'bcd', 'bad', 'cad']
#endregion

#region Задача 5
def test_sort_users():
    users = [{"name": "Ivan", "age": 20},
                {"name": "Petr", "age": 30},
                {"name": "Nata", "age": 25},
                {"name": "Serg", "age": 19},
                {"name": "Igor", "age": 31}
            ]

    asc = func.sort_users_by_age_ascending(users)
    assert asc == [{'name': 'Serg', 'age': 19}, {'name': 'Ivan', 'age': 20}, {'name': 'Nata', 'age': 25}, {'name': 'Petr', 'age': 30}, {'name': 'Igor', 'age': 31}]

    desc = func.sort_users_by_age_descending(users)
    assert desc == [{'name': 'Igor', 'age': 31}, {'name': 'Petr', 'age': 30}, {'name': 'Nata', 'age': 25}, {'name': 'Ivan', 'age': 20}, {'name': 'Serg', 'age': 19}]

#endregion



