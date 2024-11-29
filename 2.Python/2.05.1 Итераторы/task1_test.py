import pytest
from task1 import task1_list_check

@pytest.mark.parametrize("test_input, only_digit, has_positive_digit", 
                         [("5 6 2 7 8", True, True),
                          ("5 6 2 7 s", False, True),
                          ("d f", False, False),
                          ("-1 -5", True, False),
                          ("-1.2 -5.2", True, False)
                         ],)
def test_task1_list_check(test_input, only_digit, has_positive_digit):
    res = task1_list_check(test_input.split())
    assert res["only_digit"]==only_digit and res["has_positive_digit"]==has_positive_digit
