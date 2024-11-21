import math

class ComplexNumber:

    def __init__(self, alg: tuple = None, polar: tuple = None) -> None:
        if alg is not None and polar is not None or \
           alg is None and polar is None:
            raise ValueError("Необходимо указать один и только один из параметров - или alg, или polar")
        elif alg is not None:
            if type(alg) != tuple:
                raise ValueError(f"Тип параметра alg должен быть tuple")
            if len(alg) != 2:
                raise ValueError(f"В параметре alg должно быть два элемента")
            if not isinstance(alg[0], int) and not isinstance(alg[0], float) or \
               not isinstance(alg[1], int) and not isinstance(alg[1], float):
                raise ValueError(f"Элементы параметра alg должны содержать int или float")
            self.__real_part = alg[0]
            self.__imaginary_part = alg[1]
            self.__polar_r, self.__polar_theta = ComplexNumber.alg_to_polar(alg)
            self.__original = "alg"
        elif polar is not None:
            if type(polar) != tuple:
                raise ValueError(f"Тип параметра polar должен быть tuple")
            if len(polar) != 2:
                raise ValueError(f"В параметре polar должно быть два элемента")
            if not isinstance(polar[0], int) and not isinstance(polar[0], float) or \
               not isinstance(polar[1], int) and not isinstance(polar[1], float):
                raise ValueError(f"Элементы параметра polar должны содержать int или float")
            self.__polar_r = polar[0]
            self.__polar_theta = polar[1]
            self.__real_part, self.__imaginary_part = ComplexNumber.polar_to_alg(polar)
            self.__original = "polar"

        
    @classmethod
    def from_polar(cls, polar: tuple):
        return cls(polar=polar)

    @classmethod
    def from_alg(cls, alg: tuple):
        return cls(alg=alg)
        
    @property
    def real(self):
        return self.__real_part
    @property
    def imag(self):
        return self.__imaginary_part
    @property
    def r(self):
        return self.__polar_r
    @property
    def theta(self):
        return self._polar_thet

    @staticmethod
    def alg_to_polar(alg):
        """Перевод из алгебраической записи в полярную"""
        if alg[0] == 0 and alg[1] == 0:
            return (0, 0)
        mod = math.sqrt(alg[0] ** 2 + alg[1] ** 2)
        if (alg[0] + mod) == 0:
            return (mod, 0)
        phi = 2 * math.atan(alg[1] / (alg[0] + mod))
        return (mod, phi)
    @staticmethod
    def polar_to_alg(polar):
        """Перевод из полярной записи в алгебраическую"""
        real = polar[0] * math.cos(polar[1])
        imag = polar[0] * math.sin(polar[1])
        return (real, imag)
    
    @property
    def polar(self):
        return (self.__polar_r, self.__polar_theta)
    @property
    def algebraic(self):
        return (self.__real_part, self.__imaginary_part)

    @property
    def polar_as_str(self):
        return f"{round(self.__polar_r, 4)}(cos({round(self.__polar_theta, 4)})+sin({round(self.__polar_theta, 4)}))"
    @property
    def algebraic_as_str(self):
        return f"{round(self.__real_part, 4)}+{round(self.__imaginary_part, 4)}j"
    
    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(alg=(self.__real_part + other.__real_part, self.__imaginary_part + other.__imaginary_part))
        elif isinstance(other, int) or isinstance(other, float):
            return ComplexNumber(alg=(self.__real_part + other, self.__imaginary_part))
        raise TypeError("Параметр other должен быть ComplexNumber, complex, int или float")
    
    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(alg=(self.__real_part - other.__real_part, self.__imaginary_part - other.__imaginary_part))
        elif isinstance(other, int) or isinstance(other, float):
            return ComplexNumber(alg=(self.__real_part - other, self.__imaginary_part))
        raise TypeError("Параметр other должен быть ComplexNumber, complex, int или float")

    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            real = self.__real_part * other.__real_part - self.__imaginary_part * other.__imaginary_part
            imaginary = self.__real_part * other.__imaginary_part + self.__imaginary_part * other.__real_part
            return ComplexNumber(alg=(real, imaginary))
        elif isinstance(other, int) or isinstance(other, float):
            return ComplexNumber(alg=(self.__real_part * other, self.__imaginary_part * other))
        raise TypeError("Параметр other должен быть ComplexNumber, complex, int или float")

    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            znamenatel = other.__real_part ** 2 + other.__imaginary_part ** 2
            if znamenatel == 0:
                raise ZeroDivisionError("Деление на 0 невозможно")
            real_part = (self.__real_part * other.__real_part + self.__imaginary_part * other.__imaginary_part) / znamenatel
            imag_part = (other.__real_part * self.__imaginary_part - self.__real_part * other.__imaginary_part) / znamenatel
            return ComplexNumber(alg=(real_part, imag_part))
        elif isinstance(other, int) or isinstance(other, float):
            znamenatel = other ** 2
            if znamenatel == 0:
                raise ZeroDivisionError("Деление на 0 невозможно")
            real_part = (self.__real_part * other) / znamenatel
            imag_part = (other * self.__imaginary_part) / znamenatel
            return ComplexNumber(alg=(real_part, imag_part))
        raise TypeError("Параметр other должен быть ComplexNumber, complex, int или float")
    
    def __repr__(self) -> str:
        if self.__original == "polar":
            return self.polar_as_str
        else:
            return self.algebraic_as_str
    
    def __setattr__(self, name, value):
        """" Закроем свойства от изменения через c._ComplexNumber__real_part """
        if name in self.__dict__:
            raise Exception(f"Cannot change value of {name}.")
        self.__dict__[name] = value
    
if __name__ == '__main__':
    print('Демо-примеры работы с комплексными числами')
    c1 = ComplexNumber((1, 2))
    print(f'  До попытки изменения свойств объекта: {c1}')
    c1.__real_part = 1.11
    c1.__imaginary_part = 1.11
    c1.__polar_r = 1.11
    c1.__polar_theta = 1.11
    print(f'  После попытки изменения свойств объекта: {c1}')
    c2 = ComplexNumber.from_alg((3, 4))  # создание из алгебраической записи
    c3 = 5
    c4 = ComplexNumber.from_polar((2, math.pi/4)) #создание из полярной формы

    print(f'  ({c1}) + ({c2}) = {c1 + c2}')
    print(f'  ({c1}) - ({c2}) = {c1 - c2}')
    print(f'  ({c1} * {c2}) = {c1 * c2}')
    print(f'  ({c1} / {c2}) = {c1 / c2}')
    print(f'  ({c1}) + ({c3}) = {c1 + c3}')
    print(f'  ({c1}) - ({c3}) = {c1 - c3}')
    print(f'  ({c1}) * {c3} = {c1 * c3}')
    print(f'  ({c1}) / {c3} = {c1 / c3}')
    print(f"  Алгебраическая запись '{c1.algebraic_as_str}' в полярной форме '{c1.polar_as_str}'")
    print(f"  Полярная форма '{c4.polar_as_str}' в алгебраической записи '{c4.algebraic_as_str}'")

