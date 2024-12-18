def _min_max(x, y, z):
    a = sorted([x, y, z])
    return [a[2], a[0], a[1]]
def min_max(x, y, z):
    a = _min_max(x, y, z)
    print(f'{a[0]}, {a[1]}, {a[2]}', sep='\n')


def _lucky_ticket(num):
    calc_sum = lambda x: int(x[0]) + int(x[1]) + int(x[2])
    left3 = calc_sum(num[:3])
    right3 = calc_sum(num[3:])
    res = 'Счастливый' if left3==right3 else 'Обычный'
    return res
def lucky_ticket(num):
    res = _lucky_ticket(num)
    print(f'Билет {res}')
    
if  __name__ == '__main__':

    min_max(1, 2, 3)
    min_max(3, 2, 1)
    min_max(2, 3, 1)
    min_max(3, 1, 2)
    min_max(3, 3, 2)
    min_max(3, 3, 3)

    lucky_ticket("123456")
    lucky_ticket("424811")
