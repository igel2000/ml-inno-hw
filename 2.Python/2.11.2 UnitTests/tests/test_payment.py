from unittest.mock import MagicMock
from source.credit_card import CreditCard
from source.payment_form import PaymentForm
import pytest

"""
Тесты реализованы только для "проверки функциональности формы оплаты на сайте" как и написано в задании.
У класса PaymentForm() только один метод - - pay().
"""

@pytest.mark.unit
def test_payment_form_charge_success():
    """Тест успешной оплаты"""
    # тест через MagicMock
    mocked_credit_card = MagicMock(spec=CreditCard)
    payment_form = PaymentForm(mocked_credit_card)
    result = payment_form.pay(100.0)
    assert result == "Платеж успешен"

@pytest.mark.unit
def test_payment_form_charge_exception(mocker):
    """Тест ошибки при оплате"""
    # тест через mocker.patch.object()
    # "пропатчим" метод CretiCard().charge() - путь при его вызове генерируется исключение
    mocker.patch.object(CreditCard, 'charge', side_effect=[ValueError("Превышен лимит")])
    # создать замоканый экзепляр класса CreditCard
    mocked_credit_card = CreditCard()  
    # создать форму и вызвать метод оплаты
    payment_form = PaymentForm(mocked_credit_card)
    result = payment_form.pay(1500.0)
    # в результате должно быть сообщение об ошибке
    assert "Ошибка: Превышен лимит" in result
