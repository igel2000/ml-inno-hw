CREATE TABLE actor (
  actor_id numeric NOT NULL ,
  first_name VARCHAR(45) NOT NULL,
  last_name VARCHAR(45) NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (actor_id)
);
 

CREATE TABLE country (
  country_id SMALLINT NOT NULL,
  country VARCHAR(50) NOT NULL,
  last_update TIMESTAMP,
  PRIMARY KEY  (country_id)
);

CREATE TABLE city (
  city_id int NOT NULL,
  city VARCHAR(50) NOT NULL,
  country_id SMALLINT NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (city_id)
);

CREATE TABLE address (
  address_id int NOT NULL,
  address VARCHAR(50) NOT NULL,
  address2 VARCHAR(50) DEFAULT NULL,
  district VARCHAR(20) NOT NULL,
  city_id INT  NOT NULL,
  postal_code VARCHAR(10) DEFAULT NULL,
  phone VARCHAR(20) NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (address_id)
);


CREATE TABLE language (
  language_id SMALLINT NOT NULL ,
  name CHAR(20) NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY (language_id)
);


CREATE TABLE category (
  category_id SMALLINT NOT NULL,
  name VARCHAR(25) NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (category_id)
);


CREATE TABLE film_actor (
  actor_id INT NOT NULL,
  film_id  INT NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (actor_id,film_id)
);

CREATE TABLE customer (
  customer_id INT NOT NULL,
  store_id INT NOT NULL,
  first_name VARCHAR(45) NOT NULL,
  last_name VARCHAR(45) NOT NULL,
  email VARCHAR(50) DEFAULT NULL,
  address_id INT NOT NULL,
  active CHAR(1) DEFAULT 'Y' NOT NULL,
  create_date TIMESTAMP NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (customer_id)
);


CREATE TABLE film (
  film_id int NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT DEFAULT NULL,
  release_year VARCHAR(4) DEFAULT NULL,
  language_id SMALLINT NOT NULL,
  original_language_id SMALLINT DEFAULT NULL,
  rental_duration SMALLINT  DEFAULT 3 NOT NULL,
  rental_rate DECIMAL(4,2) DEFAULT 4.99 NOT NULL,
  length SMALLINT DEFAULT NULL,
  replacement_cost DECIMAL(5,2) DEFAULT 19.99 NOT NULL,
  rating VARCHAR(10) DEFAULT 'G',
  special_features VARCHAR(100) DEFAULT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (film_id)
);


CREATE TABLE film_category (
  film_id INT NOT NULL,
  category_id SMALLINT  NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY (film_id, category_id)
);

CREATE TABLE film_text (
  film_id SMALLINT NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  PRIMARY KEY  (film_id)
);

CREATE TABLE inventory (
  inventory_id INT NOT NULL,
  film_id INT NOT NULL,
  store_id INT NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (inventory_id)
);


CREATE TABLE staff (
  staff_id SMALLINT NOT NULL,
  first_name VARCHAR(45) NOT NULL,
  last_name VARCHAR(45) NOT NULL,
  address_id INT NOT NULL,
  picture bytea DEFAULT NULL,
  email VARCHAR(50) DEFAULT NULL,
  store_id INT NOT NULL,
  active SMALLINT DEFAULT 1 NOT NULL,
  username VARCHAR(16) NOT NULL,
  password VARCHAR(40) DEFAULT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (staff_id)
);

CREATE TABLE store (
  store_id INT NOT NULL,
  manager_staff_id SMALLINT NOT NULL,
  address_id INT NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (store_id)
);


CREATE TABLE payment (
  payment_id int NOT NULL,
  customer_id INT  NOT NULL,
  staff_id SMALLINT NOT NULL,
  rental_id INT DEFAULT NULL,
  amount DECIMAL(5,2) NOT NULL,
  payment_date TIMESTAMP NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY  (payment_id)
);


CREATE TABLE rental (
  rental_id INT NOT NULL,
  rental_date TIMESTAMP NOT NULL,
  inventory_id INT  NOT NULL,
  customer_id INT  NOT NULL,
  return_date TIMESTAMP DEFAULT NULL,
  staff_id SMALLINT  NOT NULL,
  last_update TIMESTAMP NOT NULL,
  PRIMARY KEY (rental_id)
);



alter table city add CONSTRAINT fk_city_country FOREIGN KEY (country_id) REFERENCES country (country_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table address add CONSTRAINT fk_address_city FOREIGN KEY (city_id) REFERENCES city (city_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table film_actor add CONSTRAINT fk_film_actor_actor FOREIGN KEY (actor_id) REFERENCES actor (actor_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table film_actor add CONSTRAINT fk_film_actor_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table customer add CONSTRAINT fk_customer_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table customer add CONSTRAINT fk_customer_address FOREIGN KEY (address_id) REFERENCES address (address_id) ON DELETE NO ACTION ON UPDATE CASCADE;

alter table film add CONSTRAINT CHECK_special_features CHECK(special_features is null or
                                                             special_features like '%Trailers%' or
                                                             special_features like '%Commentaries%' or
                                                             special_features like '%Deleted Scenes%' or
                                                             special_features like '%Behind the Scenes%');
alter table film add CONSTRAINT CHECK_special_rating CHECK(rating in ('G','PG','PG-13','R','NC-17'));
alter table film add CONSTRAINT fk_film_language FOREIGN KEY (language_id) REFERENCES language (language_id);
alter table film add CONSTRAINT fk_film_language_original FOREIGN KEY (original_language_id) REFERENCES language (language_id);


alter table film_category add CONSTRAINT fk_film_category_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table film_category add CONSTRAINT fk_film_category_category FOREIGN KEY (category_id) REFERENCES category (category_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table inventory add CONSTRAINT fk_inventory_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table inventory add CONSTRAINT fk_inventory_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table staff add CONSTRAINT fk_staff_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table staff add CONSTRAINT fk_staff_address FOREIGN KEY (address_id) REFERENCES address (address_id) ON DELETE NO ACTION ON UPDATE CASCADE;
alter table store add CONSTRAINT fk_store_address FOREIGN KEY (address_id) REFERENCES address (address_id);
alter table payment add CONSTRAINT fk_payment_rental FOREIGN KEY (rental_id) REFERENCES rental (rental_id) ON DELETE SET NULL ON UPDATE CASCADE;
alter table payment add CONSTRAINT fk_payment_customer FOREIGN KEY (customer_id) REFERENCES customer (customer_id);
alter table payment add CONSTRAINT fk_payment_staff FOREIGN KEY (staff_id) REFERENCES staff (staff_id);

alter table rental add CONSTRAINT fk_rental_staff FOREIGN KEY (staff_id) REFERENCES staff (staff_id);
alter table rental add CONSTRAINT fk_rental_inventory FOREIGN KEY (inventory_id) REFERENCES inventory (inventory_id);
alter table rental add CONSTRAINT fk_rental_customer FOREIGN KEY (customer_id) REFERENCES customer (customer_id);


CREATE  INDEX idx_actor_last_name ON actor(last_name);
CREATE  INDEX idx_fk_country_id ON city(country_id);

CREATE  INDEX idx_fk_city_id ON address(city_id);

CREATE  INDEX idx_fk_film_actor_film ON film_actor(film_id);
CREATE  INDEX idx_fk_film_actor_actor ON film_actor(actor_id);

CREATE  INDEX idx_customer_fk_store_id ON customer(store_id);
CREATE  INDEX idx_customer_fk_address_id ON customer(address_id);
CREATE  INDEX idx_customer_last_name ON customer(last_name);

CREATE  INDEX idx_fk_language_id ON film(language_id);
CREATE  INDEX idx_fk_original_language_id ON film(original_language_id);

CREATE  INDEX idx_fk_film_category_film ON film_category(film_id);
CREATE  INDEX idx_fk_film_category_category ON film_category(category_id);

CREATE  INDEX idx_fk_film_id ON inventory(film_id);
CREATE  INDEX idx_fk_film_id_store_id ON inventory(store_id,film_id);

CREATE  INDEX idx_fk_staff_store_id ON staff(store_id);
CREATE  INDEX idx_fk_staff_address_id ON staff(address_id);

CREATE  INDEX idx_store_fk_manager_staff_id ON store(manager_staff_id);
CREATE  INDEX idx_fk_store_address ON store(address_id);

CREATE  INDEX idx_fk_staff_id ON payment(staff_id);
CREATE  INDEX idx_fk_customer_id ON payment(customer_id);

CREATE INDEX idx_rental_fk_inventory_id ON rental(inventory_id);
CREATE INDEX idx_rental_fk_customer_id ON rental(customer_id);
CREATE INDEX idx_rental_fk_staff_id ON rental(staff_id);
CREATE UNIQUE INDEX   idx_rental_uq  ON rental (rental_date,inventory_id,customer_id);


