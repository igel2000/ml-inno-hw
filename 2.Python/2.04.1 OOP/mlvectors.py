from typing import List
import math

class MLVector(List):
    
    __abs_tol = 1e-8
    
    def __init__(self, coords: List | tuple):
        if isinstance(coords, (list, tuple)):
            if len(coords) == 0:
                raise ValueError("Вектор не может быть пустым")
            if all(isinstance(i, (int, float)) for i in coords):
                self.__coords = list(coords)
            else:
                raise ValueError(f"Все координаты вектора должны быть int или float: {coords}")
        elif isinstance(coords, (int, float)):
            self.__coords = [coords]
        else:
            raise ValueError(f"Неподдерживаемый тип данных {type(coords)}")
    @property
    def coords(self):
        return self.__coords
    @property
    def norma_e(self):
        return math.sqrt(sum( i*i for i in self.__coords))
    @property
    def n(self):
        return len(self.coords)
    
    def __repr__(self):
        return f"({",".join(map(str,self.coords))})"
    
    @staticmethod
    def cosa(vector_a, vector_b):
        if vector_a.n != vector_b.n:
            raise ValueError("Вектора должны быть одной размерности")
        if vector_a.norma_e ==0 or vector_b.norma_e == 0:
            raise ValueError("Вектора должны быть не нулевой длины")
        return (vector_a * vector_b) / (vector_a.norma_e * vector_b.norma_e)
                  
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_coords = [ self.__coords[i] * other for i in range(len(self.__coords))]
            return MLVector(new_coords)
        elif isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            return sum(self.__coords[i]*other.__coords[i] for i in range(len(self.__coords)))
        else:
            raise ValueError("Умножать можно на MLVector, int или float")
    
    def __add__(self, other):
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            new_coords = [ self.__coords[i] + other.__coords[i] for i in range(len(self.__coords))]
            return MLVector(new_coords)
        else:
            raise ValueError("Складывать можно только с MLVector")
    
    def __sub__(self, other):
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            new_coords = [ self.__coords[i] - other.__coords[i] for i in range(len(self.__coords))]
            return MLVector(new_coords)
        else:
            raise ValueError("Вычитать можно только MLVector")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            new_coords = [ self.__coords[i] / other for i in range(len(self.__coords))]
            return MLVector(new_coords)
        else:
            raise ValueError("Делить можно на int или float")
        
    def __eq__(self, other: object) -> bool:
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            for i in range(len(self.__coords)):
                if not math.isclose(self.__coords[i], other.__coords[i], abs_tol=self.__abs_tol):
                    return False
            return True
        else:
            raise ValueError("Сравнивать можно только с MLVector")

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            return self.norma_e < other.norma_e
        else:
            raise ValueError("Сравнивать можно только с MLVector")

    def __le__(self, other: object) -> bool:
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            return self.norma_e <= other.norma_e
        else:
            raise ValueError("Сравнивать можно только с MLVector")

    def __gt__(self, other: object) -> bool:
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            return self.norma_e > other.norma_e
        else:
            raise ValueError("Сравнивать можно только с MLVector")

    def __ge__(self, other: object) -> bool:
        if isinstance(other, MLVector):
            if self.n != other.n:
                raise ValueError("Вектора должны быть одной размерности")
            return self.norma_e >= other.norma_e
        else:
            raise ValueError("Сравнивать можно только с MLVector")

    def __setattr__(self, name, value):
        """" Закроем свойства от изменения через c._MLVector__real_part """
        if name in self.__dict__:
            raise Exception(f"Cannot change value of {name}.")
        self.__dict__[name] = value
        

if __name__ == "__main__":
    pass