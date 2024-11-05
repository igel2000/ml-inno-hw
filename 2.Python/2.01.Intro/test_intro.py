import pytest
import intro

def test_min_max_1_2_3():
    assert intro._min_max(1, 2, 3) == [3, 1, 2]
    assert intro._min_max(3, 2, 1) == [3, 1, 2]
    assert intro._min_max(2, 3, 1) == [3, 1, 2]
    assert intro._min_max(3, 1, 2) == [3, 1, 2]
    assert intro._min_max(3, 3, 2) == [3, 2, 3]
    assert intro._min_max(3, 3, 3) == [3, 3, 3]

def test_lucky_ticket_lucky():
    assert intro._lucky_ticket("424811") == "Счастливый"
    assert intro._lucky_ticket("600006") == "Счастливый"
    assert intro._lucky_ticket("222060") == "Счастливый"
def test_lucky_ticket_unlucky():
    assert intro._lucky_ticket("123456") == "Обычный"
    assert intro._lucky_ticket("654321") == "Обычный"

    