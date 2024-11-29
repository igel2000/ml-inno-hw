def task1_list_check(l):
    def is_num(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    res = {}
    res["only_digit"] = all(is_num(c) for c in l)
    res["has_positive_digit"] = any(is_num(c) and float(c) > 0 for c in l)
    return res

if __name__ == "__main__":
    data = input("Введите список значений через пробелы:")
    l = data.split(" ")
    res = task1_list_check(l)
    print(f"Отсортированный список: {sorted(l)}")
    print("  В списке только числа" if res["only_digit"] else "В списке есть нечисла")
    print("  В списке есть положительное число" if res["has_positive_digit"] else "В списке нет положительного числа")
