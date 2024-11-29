import pytest
from task3 import password_generator

@pytest.mark.parametrize('seed,password_length,expected', [(42, None, 'odJFCrnl2edl'),
                                                           (None, None, 'BD@dz1C5Jau2'),
                                                           (None, 5, 'RJtBR')])
def test_password_generator(seed, password_length, expected):
    p = ""
    if password_length is None:
        p  = password_generator(seed=seed).__next__()
    else:
        p = password_generator(seed=seed, password_length=password_length).__next__()
    assert p == expected
