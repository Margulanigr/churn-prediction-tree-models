--1 churn rate
select country, count(*), sum(churn), round(avg(churn) *100,2) as churn_rate
from bank_churn
group by country;

--2 age group
select count(*), round(avg(churn) *100, 2) as churn_rate,
	case
		when age<=30 then '0-30'
		when age<=40 then '31-40'
		when age<=50 then '41-50'
		when age<=60 then '51-60'
		else '60+'
	end as age_group
from bank_churn
group by age_group;


--3 number of products
select count(*), products_number, round(avg(churn) *100, 2) as churn_rate
from employees
group by products_number
having count(*) > 100
order by products_number;

--4
select country, count(*), round(avg(churn) * 100, 2), (case when balance = 0 then 'zero balance' else 'non-zero balance' end) as balancee
from bank_churn
group by country, case when balance = 0 then 'zero balance' else 'non-zero balance' end;

--5 top 3 clients by balance in each country
with ranked_clients as
    (select customer_id, balance, country, ROW_NUMBER() OVER(PARTITION by country order by balance desc) as rn
    from bank_churn)
select customer_id, country, balance from ranked_clients
where rn<=3;

-- 6 Outflow for each country and its deviation from the total outflow for the bank
with bank_total as
    (select round(avg(churn)*100,2 ) as total_churn_pct
    from bank_churn),
country_stats as
    (select country, count(*) as customer_count, round(avg(churn)*100,2) as country_churn_pct
    from bank_churn
    group by country)
select c.country, c.customer_count, c.country_churn_pct, b.total_churn_pct, round(c.country_churn_pct - b.total_churn_pct, 2) as diff_from_bank_pct
from country_stats c cross join bank_total b
order by country_churn_pct desc;