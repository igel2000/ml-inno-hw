import pytest
import cmath
import math
from complex_number import ComplexNumber

complex_numbers = [(3,  4),
                   (-8, 1),
                   (3,  0),
                   (0,  2),
                   (2**0.5, 2**0.5),
                   (0,  0)]

real_numbers = [-2.5, -1, 0, 1,  2.5]

#region service_functions
""" Корректность работы класса ComplexNumber проверяем через выполнение аналогичных действий с классом cmath.complex """
def check_complex(my_complex, python_complex):
    return math.isclose(my_complex.real, python_complex.real, abs_tol=0.0000001) and math.isclose(my_complex.imag, python_complex.imag, abs_tol=0.0000001)
def gen_comlex(params):
    my_complex = ComplexNumber(params)
    python_comlpex= complex(params[0], params[1])
    return my_complex, python_comlpex
#endregion

#region create_tests
def test_ComplexNumber_create_positive():
    """Проверка создания комплексных чисел"""
    assert type(ComplexNumber((1,2))) == ComplexNumber
    assert type(ComplexNumber((1.5,2.5))) == ComplexNumber

    for alg_tuple in complex_numbers:
        c1_test = complex(alg_tuple[0], alg_tuple[1])
        c1_from_alg = ComplexNumber.from_alg(alg_tuple)
        c1_from_polar = ComplexNumber.from_polar(cmath.polar(c1_test))
        assert check_complex(c1_from_alg, c1_test)
        assert check_complex(c1_from_alg, c1_from_polar)
        assert check_complex(c1_from_polar, c1_test)

def test_ComplexNumber_create_negative():
    """Проверка обработки ошибок при создании комплексных чисел"""
    with pytest.raises(ValueError):
        ComplexNumber(1)
    with pytest.raises(ValueError):
        ComplexNumber(("1", "2"))
    with pytest.raises(ValueError):
        ComplexNumber(("a", "b"))
    with pytest.raises(ValueError):
        ComplexNumber((1))
    with pytest.raises(ValueError):
        ComplexNumber((1,2,3))
    with pytest.raises(ValueError):
        ComplexNumber(alg=None, polar=None)
    with pytest.raises(ValueError):
        ComplexNumber(alg=(1,2), polar=(2,1))
    
def test_ComplexNumber_repr():
    """Проверка корректности преобразования в строку"""
    
    assert ComplexNumber((1,2)).__repr__() == "1+2j"
    assert ComplexNumber.from_alg((1,2)).__repr__() == "1+2j"
    assert ComplexNumber((1.5,2.5)).__repr__() == "1.5+2.5j"
    assert ComplexNumber.from_alg((1.5,2.5)).__repr__() == "1.5+2.5j"
    
#endregion    

#region add_tests
def test_ComplexNumber_add_positive():
    for alg_tuple1 in complex_numbers:
        # сложение двух комплексных чисел с перестановкой
        for alg_tuple2 in complex_numbers:
            c1, c1_t = gen_comlex(alg_tuple1)
            c2, c2_t = gen_comlex(alg_tuple2)
            c3 = c1 + c2
            c3_t = c1_t + c2_t
            assert check_complex(c3, c3_t)
            c3 = c2 + c1
            c3_t = c2_t + c1_t
            assert check_complex(c3, c3_t)
        # сложение с обычным числом
        for real_num in real_numbers:
            c3 = c1 + real_num
            c3_t = c1_t + real_num
            assert check_complex(c3, c3_t)

def test_ComplexNumber_add_negaive():
    c1 = ComplexNumber((1,2))
    c2 = "str"
    with pytest.raises(TypeError):
        c3 = c1 + c2
#endregion    

#region sub_tests
def test_ComplexNumber_sub_positive():
    for alg_tuple1 in complex_numbers:
        # вычитание двух комплексных чисел
        for alg_tuple2 in complex_numbers:
            c1, c1_t = gen_comlex(alg_tuple1)
            c2, c2_t = gen_comlex(alg_tuple2)
            c3 = c1 - c2
            c3_t = c1_t - c2_t
            assert check_complex(c3, c3_t)
        # вычитание обычного числа
        for real_num in real_numbers:
            c3 = c1 - real_num
            c3_t = c1_t - real_num
            assert check_complex(c3, c3_t)        

def test_ComplexNumber_sub_negaive():
    c1 = ComplexNumber((1,2))
    c2 = "str"
    with pytest.raises(TypeError):
        c3 = c1 - c2

#endregion    

#region mul_tests
def test_ComplexNumber_mul_positive():
    for alg_tuple1 in complex_numbers:
        # умножение двух комплексных чисел с перестановкой
        for alg_tuple2 in complex_numbers:
            c1, c1_t = gen_comlex(alg_tuple1)
            c2, c2_t = gen_comlex(alg_tuple2)
            c3 = c1 * c2
            c3_t = c1_t * c2_t
            assert check_complex(c3, c3_t)
            c3 = c2 * c1
            c3_t = c2_t * c1_t
            assert check_complex(c3, c3_t)
        # умножение на обычные число
        for real_num in real_numbers:
            c3 = c1 * real_num
            c3_t = c1_t * real_num
            assert check_complex(c3, c3_t)    

def test_ComplexNumber_mul_negaive():
    c1 = ComplexNumber((1,2))
    c2 = "str"
    with pytest.raises(TypeError):
        c3 = c1 * c2

#endregion    

#region div_tests
def test_ComplexNumber_div_positive():
    for alg_tuple1 in complex_numbers:
        # деление на комплексное число
        for alg_tuple2 in complex_numbers:
            if (alg_tuple1[0] != 0) and (alg_tuple2[0] != 0):
                c1, c1_t = gen_comlex(alg_tuple1)
                c2, c2_t = gen_comlex(alg_tuple2)
                c3 = c1 / c2
                c3_t = c1_t / c2_t
                assert check_complex(c3, c3_t)
        # деление на обычное число
        for real_num in real_numbers:
            if real_num != 0:
                c3 = c1 / real_num
                c3_t = c1_t / real_num
                assert check_complex(c3, c3_t)        

def test_ComplexNumber_div_negaive():
    c0 = ComplexNumber((0,0))
    c1 = ComplexNumber((1,2))
    c2 = "str"
    with pytest.raises(TypeError):
        c3 = c1 / c2
    with pytest.raises(ZeroDivisionError):
        c3 = c1 / c0
        
#endregion    

#region convert_tests
def test_ComplexNumber_convert():
    """Проверка конвертации комплексных чисел из алгебраической формы в полярную и обратно"""
    for alg_tuple in complex_numbers:
        c1_test = complex(alg_tuple[0], alg_tuple[1])
        c1_from_alg = ComplexNumber.from_alg(alg_tuple)
        c1_from_alg_polar = ComplexNumber.from_polar(c1_from_alg.polar)
        c1_from_polar = ComplexNumber.from_polar(cmath.polar(c1_test))
        c1_from_polar_alg = ComplexNumber.from_alg(c1_from_polar.algebraic)
        assert check_complex(c1_from_alg, c1_from_alg_polar)
        assert check_complex(c1_from_polar, c1_from_polar_alg)

#endregion

#region  immutable_tests
def test_ComplexNumber_immutable():
    c1 = ComplexNumber((1, 2))
    with pytest.raises(Exception):
        c1._ComplexNumber__real_part = 2.22
#endregion



