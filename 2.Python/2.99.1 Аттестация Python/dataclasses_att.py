#from dataclasses import dataclass
#import datetime
#from types import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, BLOB, NUMERIC, TEXT # ORM 
from sqlalchemy.orm import DeclarativeBase # базовый класс для создания декларативных моделей SQLAlchemy


class BaseTable(DeclarativeBase):
    pass

class Actor(BaseTable):
    __tablename__ = 'actor' # имя таблицы
    actor_id = Column(Integer, primary_key=True)
    first_name = Column(String(45))
    last_name = Column(String(45))
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Actor(actor_id={self.actor_id}, first_name={self.first_name}, last_name={self.last_name})>'
 

class Country(BaseTable):
    __tablename__ = 'country' # имя таблицы
    country_id = Column(Integer, primary_key=True)
    country = Column(String(50))
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Country(country_id={self.country_id}, country={self.country})>'

class City(BaseTable):
    __tablename__ = 'city' # имя таблицы
    city_id = Column(Integer, primary_key=True)
    city = Column(String(50))
    country_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<City(city_id={self.city_id}, city={self.city})>'


class Address(BaseTable):
    __tablename__ = 'address' # имя таблицы
    address_id = Column(Integer, primary_key=True)
    address = Column(String(50))
    address2 = Column(String(50), nullable = True)
    district = Column(String(20))
    city_id = Column(Integer)
    postal_code = Column(String(10), nullable = True)
    phone = Column(String(20))
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Address(address_id={self.address_id}, address={self.address})>'

class Language(BaseTable):
    __tablename__ = 'language' # имя таблицы
    language_id = Column(Integer, primary_key=True)
    name = Column(String(20))
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Language(language_id={self.language_id}, name={self.name})>'

class Category(BaseTable):
    __tablename__ = 'category' # имя таблицы
    category_id = Column(Integer, primary_key=True)
    name = Column(String(25))
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Category(category_id={self.category_id}, name={self.name})>'

class FilmActor(BaseTable):
    __tablename__ = 'film_actor' # имя таблицы
    actor_id = Column(Integer, primary_key=True)
    film_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<FilmActor(actor_id={self.actor_id}, film_id={self.film_id})>'

class Customer(BaseTable):
    __tablename__ = 'customer' # имя таблицы
    customer_id = Column(Integer, primary_key=True)
    store_id = Column(Integer)
    first_name = Column(String(45))
    last_name = Column(String(45))
    email = Column(String(50), nullable = True)
    address_id = Column(Integer)
    active = Column(String(1))
    create_date = Column(DateTime)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Customer(customer_id={self.customer_id}, first_name={self.first_name}, last_name={self.last_name})>'


class Film(BaseTable):
    __tablename__ = 'film' # имя таблицы
    film_id = Column(Integer, primary_key=True)
    title = Column(String(255))
    description = Column(TEXT, nullable = True)
    release_year = Column(String(4), nullable = True)
    language_id = Column(Integer)
    original_language_id = Column(Integer, nullable=True)
    rental_duration = Column(Integer)
    rental_rate = Column(NUMERIC(4,2))
    length = Column(Integer, nullable=True)
    replacement_cost = Column(NUMERIC(5,2))
    rating = Column(String(10))
    special_features = Column(String(100), nullable = True)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Film(film_id={self.film_id}, title={self.title})>'


class FilmCategory(BaseTable):
    __tablename__ = 'film_category' # имя таблицы
    film_id = Column(Integer, primary_key=True)
    category_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<FilmCategory(film_id={self.film_id}, category_id={self.category_id})>'

class FilmText(BaseTable):
    __tablename__ = 'film_text' # имя таблицы
    film_id = Column(Integer, primary_key=True)
    title = Column(String(255))
    description = Column(String())
    def __repr__(self):
        return f'<FilmText(film_id={self.film_id}, title={self.title})>'

class Inventory(BaseTable):
    __tablename__ = 'inventory' # имя таблицы
    inventory_id = Column(Integer, primary_key=True)
    film_id = Column(Integer)
    store_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Inventory(inventory_id={self.inventory_id}, film_id={self.film_id}, store_id={self.store_id})>'

class Staff(BaseTable):
    __tablename__ = 'staff' # имя таблицы
    staff_id = Column(Integer, primary_key=True) 
    first_name = Column(String(45))
    last_name = Column(String(45))
    address_id = Column(Integer)
    picture = Column(BLOB)
    email = Column(String(50), nullable = True)
    store_id = Column(Integer)
    active = Column(Integer)
    username = Column(String(16))
    password = Column(String(40), nullable = True)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Staff(staff_id={self.staff_id}, first_name={self.first_name}, last_name={self.last_name})>'

class Store(BaseTable):
    __tablename__ = 'store' # имя таблицы
    store_id = Column(Integer, primary_key=True)
    manager_staff_id = Column(Integer)
    address_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Store(store_id={self.store_id}, manager_staff_id={self.manager_staff_id}, address_id={self.address_id})>'

class Payment(BaseTable):
    __tablename__ = 'payment' # имя таблицы
    payment_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer)
    staff_id = Column(Integer)
    rental_id = Column(Integer, nullable=True)
    amount = Column(NUMERIC(5,2))
    payment_date = Column(DateTime)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Payment(payment_id={self.payment_id}, customer_id={self.customer_id}, staff_id={self.staff_id}, rental_id={self.rental_id}, amount={self.amount}, payment_date={self.payment_date})>'

class Rental(BaseTable):
    __tablename__ = 'rental' # имя таблицы
    rental_id = Column(Integer, primary_key=True)
    rental_date = Column(DateTime)
    inventory_id = Column(Integer)
    customer_id = Column(Integer)
    return_date = Column(DateTime, nullable=True)
    staff_id = Column(Integer)
    last_update = Column(DateTime, nullable=True)
    def __repr__(self):
        return f'<Rental(rental_id={self.rental_id}, rental_date={self.rental_date}, inventory_id={self.inventory_id}, customer_id={self.customer_id}, return_date={self.return_date}, staff_id={self.staff_id})>'
