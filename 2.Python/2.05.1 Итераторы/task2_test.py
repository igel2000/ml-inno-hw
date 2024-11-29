import pytest
from task2 import CyclicIterator

@pytest.mark.parametrize("lst_template, repeat_count, expected", 
                         [([1, 2, 3], 2, [1, 2, 3, 1, 2, 3]),
                          ([-1, 2], None, [-1, 2, -1, 2, -1, 2, -1, 2, -1, 2])
                         ])
def test_cyclic_iterator(lst_template, repeat_count, expected):
    lst = []
    if repeat_count is None:
        lst = [l for l in CyclicIterator(obj=lst_template)]
    else:
        lst = [l for l in CyclicIterator(obj=lst_template, max_repeat=repeat_count)]
    assert lst == expected


def test_cyclic_iterator_break():
    lst = []
    for l in CyclicIterator(["a", "b"]):
        lst.append(l)
        break
    assert lst == ["a"]
