"""
Реализации класса нет, т.к. по условиям задачи всё равно будем мокать 
и тесты нужны на проверку функциональности формы оплаты на сайте, а не этого класса.

По сути моделируем ситуацию, когда у нас сделан только прототип класса, который мы собираемся использовать не дожидаясь реализации.
"""

class CreditCard:
    """Класс Кредитная карта."""
    
    def get_card_number(self):
        raise NotImplementedError("Метод get_card_number() не реализован")

    def get_card_holder(self):
        raise NotImplementedError("Метод get_card_holder() не реализован")

    def get_expiry_date(self):
        raise NotImplementedError("Метод get_expiry_date() не реализован")

    def get_cvv(self):
        raise NotImplementedError("Метод get_cvv() не реализован")

    def charge(self, amount):
        raise NotImplementedError("Метод charge() не реализован")
