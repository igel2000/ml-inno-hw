from source.credit_card import CreditCard
class PaymentForm:
    """Класс для формы оплаты"""
    def __init__(self, credit_card: CreditCard):
        self.credit_card = credit_card

    def pay(self, amount):
        try:
            self.credit_card.charge(amount)
            return "Платеж успешен"
        except ValueError as e:
            return f"Ошибка: {e}"
