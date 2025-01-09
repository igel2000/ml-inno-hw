-- 1) Напишите запрос, чтобы получить все названия продуктов и соответствующие им торговые марки (brand).
select p.product_name, b.brand_name 
  from products as p
  left outer join brands b on p.brand_id = b.brand_id 
  order by 1,2

-- 2) Напишите запрос, чтобы найти всех активных сотрудников и наименования магазинов, в которых они работают.

--Для проверики: Добавить сотрудника без магазина
--insert into staffs
--  values (6000, 'Сотрудник', 'Без магазина', '1@1.1', '912', 1)
  
select staffs.first_name, staffs.last_name, stores.store_name 
  from staffs
  left outer join stores on staffs.store_id = stores.store_id 
  where staffs.active = 1
  order by 3, 1, 2
  
  
-- 3) Напишите запрос, чтобы перечислить всех покупателей выбранного магазина с указанием их полных имен, 
--    электронной почты и номера телефон
select customers.first_name, customers.last_name, customers.email, customers.phone, orders.store_id 
	from orders 
	join customers on orders.customer_id = customers.customer_id 
	where orders.store_id = 2
	order by 1, 2

	
-- 4) Напишите запрос для подсчета количества продуктов в каждой категории.
	
--Для проверки: добавить категорию, у которой не будет продуктов	
--insert into categories
--  values(10, 'Категория без продуктов')
	
select c.category_name, count(*)
  from products p 
  join categories c on c.category_id = p.category_id 
  group by c.category_name  
union
select c2.category_name, 0
  from categories c2 
  where not exists (select 1 from products p2 where c2.category_id = p2.category_id)

  
-- 5) Напишите запрос, чтобы указать общее количество заказов для каждого клиента.

--Для проверки: Добавить заказчика без заказов  
--insert into customers 
--  values(2000, 'Заказчик', 'Пустой', '3333', '1@1.1')
  
select c.customer_id, count(*) as "orders_count"
  from customers c 
  join orders o on c.customer_id = o.customer_id 
  group by c.customer_id 
union 
select c2.customer_id, 0 as "orders_count"
  from customers c2 
  where not exists(select 1 from orders o2 where c2.customer_id = o2.customer_id)
order by 2, 1

  
-- 6) Напишите запрос, в котором будет указана информация о полном имени и общем количестве заказов клиентов, 
--    которые хотя бы 1 раз сделали заказ.
select c.first_name, c.last_name, count(*) as "orders_count"
  from customers c 
  join orders o on c.customer_id = o.customer_id 
  group by c.first_name, c.last_name
order by 3, 1, 2

    
    
    
    