import pytest
from mlvectors import MLVector
import math


#region create
def test_MLVector_create_positive():
    vectors_desc = [1, 1.5, (1), [1], (1,2), [1.2, 5], (1, 2, 2, 3, 4), [1.1, 2.2, 3.3, 4.3]]
    assert type(MLVector(vectors_desc[0])) == MLVector
    for vd in vectors_desc:
        vector = MLVector(vd)
        assert isinstance(vector.coords, list)

def test_MLVector_create_negative():
    vectors_desc = [[], (), "str", (1, "str"), [2, "ss"]]
    for vd in vectors_desc:
        with pytest.raises(ValueError):
            vector = MLVector(vd)
#endregion

#region  properties
def test_MLVector_length_positive():
    vector_desc = [([0], 0),
                   ([1], 1),
                   ([3], 3),
                   ([-1], 1),
                   ([-2], 2),
                   ((1, 1), 2**0.5),
                   ((0, 0), 0),
                   ([1, -2], 5**0.5),
                   ([-3, -4], 25**0.5),
                   ([0, 0, 0, 0], 0),
                   ([1, 1, 1, 1], 4**0.5),
                   ([-2, 1, -3, 4], 30**0.5)
                ]
    for vd in vector_desc:
        vector = MLVector(vd[0])
        assert math.isclose(vector.norma_e, vd[1], abs_tol=0.00000001)
#endregion

#region  cosinus
def test_MLVector_cosa_positive():
    vector_pairs = [((3,4), (4,3), 0.96),
                    ((7,1), (5,5), 0.8),
                    ((3,4,0), (4,4,2), 14/15),
                    ((1,0,3), (5,5,0), 0.1*(5**0.5)),
                    ]
    for vector_pair in vector_pairs:
        v1 = MLVector(vector_pair[0])
        v2 = MLVector(vector_pair[1])
        assert math.isclose(MLVector.cosa(v1, v2), vector_pair[2], abs_tol=0.00000001)
        
def test_MLVector_cosa_negative():
    vector_pairs = [((1), (1,2)),
                     ((0,0),(0,0)),
                     ((0),(0)),
                     ((0,0),(0,1))
                   ]
    for vector_pair in vector_pairs:
        with pytest.raises(ValueError):
            v1 = MLVector(vector_pair[0])
            v2 = MLVector(vector_pair[1])
            MLVector.cosa(v1, v2)
        
#endregion

#region  add
def test_MLVector_add_positive():
    vector_pairs = [((3,4), (4,4), [7, 8]),
                    ((-7,-1), (-5,-5), [-12, -6]),
                    ((3,4,0), (4,4,2), [7,8,2]),
                    ((0,0,0), (0,0,0), [0,0,0]),
                    ((0,0,0), (1,1,1), [1,1,1]),
                    ((2), (-3), [-1]),
                    ((6, 4, 11, 14, 99), (3, -2, 10, -10, 1), [9, 2, 21, 4, 100])
                    ]
    for vector_pair in vector_pairs:
        v1 = MLVector(vector_pair[0])
        v2 = MLVector(vector_pair[1])
        assert (v1 + v2) == MLVector(vector_pair[2])
        
def test_MLVector_add_negative():
    vector_pairs = [((1), (1,2))
                   ]
    for vector_pair in vector_pairs:
        with pytest.raises(ValueError):
            v1 = MLVector(vector_pair[0])
            v2 = MLVector(vector_pair[1])
            v = v1 + v2
        
    with pytest.raises(ValueError):
        v = v1 + 1
#endregion

#region  sub
def test_MLVector_sub_positive():
    vector_pairs = [((3,4), (4,4), [-1, 0]),
                    ((-7,-1), (-5,-5), [-2, 4]),
                    ((3,4,0), (4,4,2), [-1,0,-2]),
                    ((0,0,0), (0,0,0), [0,0,0]),
                    ((0,0,0), (1,1,1), [-1,-1,-1]),
                    ((1,1,1), (0,0,0), [1,1,1]),
                    ((2), (-3), [5]),
                    ((6, 4), (3, -2), [3, 6])
                    ]
    for vector_pair in vector_pairs:
        v1 = MLVector(vector_pair[0])
        v2 = MLVector(vector_pair[1])
        assert (v1 - v2) == MLVector(vector_pair[2])
        
def test_MLVector_sub_negative():
    vector_pairs = [((1), (1,2))
                   ]
    for vector_pair in vector_pairs:
        with pytest.raises(ValueError):
            v1 = MLVector(vector_pair[0])
            v2 = MLVector(vector_pair[1])
            v = v1 - v2
        
    with pytest.raises(ValueError):
        v = v1 - 1
#endregion

#region  mul
def test_MLVector_mul_positive():
    vector_pairs = [((2,-5), (-1,0), -2),
                    ((-1,2,-5), (3,-3,-3), 6),
                    ((1,2,-4), (6,-1,1), 0),
                    ((-5,-5), (-7,5), 10),
                    ((2,-5), 3, [6, -15])
                    ]
    for vector_pair in vector_pairs:
        v1 = MLVector(vector_pair[0])
        if isinstance(vector_pair[1], (int | float)):
            assert (v1 * vector_pair[1]) == MLVector(vector_pair[2])
        else:
            v2 = MLVector(vector_pair[1])
            res = v1 * v2
            assert math.isclose(res, vector_pair[2], abs_tol=0.00000001)
        
def test_MLVector_mul_negative():
    with pytest.raises(ValueError):
        v = MLVector((1,2)) * MLVector((1,2,3))
    with pytest.raises(ValueError):
        v = MLVector((1,2)) * "str"
#endregion

#region  div
def test_MLVector_div_positive():
    vector_pairs = [((2,-5), -1, [-2,5]),
                    ((10,20,30), 2.0, [5.0, 10.0, 15.0]),
                    ]
    for vector_pair in vector_pairs:
        v1 = MLVector(vector_pair[0])
        assert v1 / vector_pair[1] == MLVector(vector_pair[2])
        
def test_MLVector_div_negative():
    with pytest.raises(ValueError):
        v = MLVector((1,2)) / MLVector((1,2))
    with pytest.raises(ValueError):
        v = MLVector((1,2)) / "str"
    with pytest.raises(ZeroDivisionError):
        v = MLVector((1,2)) / 0
#endregion

#region  compare
def test_MLVector_eq_positive():
    # __eq__
    assert MLVector((-2,5)) == MLVector((-2.0   , 5.0))
    assert MLVector((-2,5)) == MLVector((-6.0/3 , 2.5*2))
    assert not MLVector((2,-5)) == MLVector((0.0   , 5))
    
    # __ne__
    assert MLVector((2,-5)) != MLVector((0.0   , 5))
    
    # __lt__
    assert MLVector((0,5)) < MLVector((0,6))

    # __le__
    assert MLVector((0,5)) <= MLVector((0,6))
    assert MLVector((0,5)) <= MLVector((0,5))

    # __gt__
    assert MLVector((0,7)) > MLVector((0,6))

    # __ge__
    assert MLVector((0,7)) >= MLVector((0,6))
    assert MLVector((0,7)) >= MLVector((0,5))
    
        
def test_MLVector_eq_negative():
    # __eq__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) == MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) == [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) == 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) == "str"

    # __ne__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) != MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) != [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) != 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) != "str"

    # __gt__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) > MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) > [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) > 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) > "str"

    # __ge__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) >= MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) >= [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) >= 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) >= "str"

    # __lt__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) < MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) < [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) < 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) < "str"

    # __le__
    with pytest.raises(ValueError):
        x = MLVector((1,2)) <= MLVector((1,2,3))
    with pytest.raises(ValueError):
        x = MLVector((1,2)) <= [1,2]
    with pytest.raises(ValueError):
        x = MLVector((1,2)) <= 1
    with pytest.raises(ValueError):
        x = MLVector((1,2)) <= "str"

#endregion

#region tolerance
def test_tolerances():
    assert MLVector.get_tolerance() == 1e-8
    #MLVector((-6.0/3 , 2.5*2))
    MLVector.set_tolerance(1e-4)
    assert MLVector.get_tolerance() == 1e-4
    v1 = MLVector((2,2))     # длина 2.8284271247461903
    v2 = MLVector((2,2.001)) # длина 2.829134319893631
    # при точности сранения 1e-4 вектора должны быть не равны
    assert v1 != v2
    # после огрубления точности сранения до 1e-2 вектора должны быть равны
    MLVector.set_tolerance(1e-2)
    assert v1 == v2
#endregion