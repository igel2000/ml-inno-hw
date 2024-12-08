from unittest import TestCase
from unittest_parametrize import parametrize
from unittest_parametrize import ParametrizedTestCase
import func

#region Задача 1
class TestTask1(ParametrizedTestCase):
    @parametrize("s, expected", [(None, False),
                                 ("", False),
                                 ("s", False),
                                 ("ss", False),
                                 ("sss", True),
                                 ("ssss", True)
                                ])
    def test_three_or_more(self, s, expected):
        assert func.three_or_more(s) == expected
#endregion

#region Задача 2
class TestTask2(ParametrizedTestCase):
    @parametrize("l, expected", [(None, []),
                                 ("", []),
                                 ("1", list([1])),
                                 ("1 2", [1, 2]),
                                 ("1 2 3", [1, 2, 3])
                                ])
    def test_get_list(self, l, expected):
        assert func.get_list(l) == expected
    
    @parametrize("l, expected", [("1 2 3", [1, 2, 3]),
                                 ("3 2 1", [1, 2, 3])
                                ])
    def test_sort_func(self, l, expected):
        assert func.sort_func(func.get_list, l) == expected
#endregion

#region Задача 3
class TestTask3(ParametrizedTestCase):
    @parametrize("s, n, expected", [("bca", 2, ['ab', 'ac', 'ba', 'bc', 'ca', 'cb'])])
    def test_get_permutations(self, s, n, expected):
        assert func.get_permutations(s, n) == expected
#endregion

#region Задача 4
class TestTask4(ParametrizedTestCase):
    @parametrize("s, k, expected", [("dbcad", 3, 
                                     ['d', 'b', 'c', 'a', 'd', 'db', 'dc', 'da', 'dd', 'bc', 
                                      'ba', 'bd', 'ca', 'cd', 'ad', 'dbc', 'dba', 'dbd', 
                                      'dca', 'dcd', 'dad', 'bca', 'bcd', 'bad', 'cad'
                                      ])])
    def test_get_combinations(self, s, k , expected):
        assert func.get_combinations(s, k) == expected
#endregion

#region Задача 5
class TestTask5(TestCase):
    def setUp(self):
        self.users = [{"name": "Ivan", "age": 20},
                      {"name": "Petr", "age": 30},
                      {"name": "Nata", "age": 25},
                      {"name": "Serg", "age": 19},
                      {"name": "Igor", "age": 31}
                     ]
        
    def test_sort_users_by_age_ascending(self):
        asc = func.sort_users_by_age_ascending(self.users)
        assert asc == [{'name': 'Serg', 'age': 19}, {'name': 'Ivan', 'age': 20}, {'name': 'Nata', 'age': 25}, {'name': 'Petr', 'age': 30}, {'name': 'Igor', 'age': 31}]

    def sort_users_by_age_descending(self):
        desc = func.sort_users_by_age_descending(self.users)
        assert desc == [{'name': 'Igor', 'age': 31}, {'name': 'Petr', 'age': 30}, {'name': 'Nata', 'age': 25}, {'name': 'Ivan', 'age': 20}, {'name': 'Serg', 'age': 19}]

#endregion
