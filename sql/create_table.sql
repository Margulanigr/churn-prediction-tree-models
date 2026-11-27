 CREATE TABLE bank_churn (
     customer_id BIGINT PRIMARY KEY,
     credit_score INTEGER,
     country VARCHAR(50),
     gender VARCHAR(20),
     age SMALLINT,
     tenure SMALLINT,
     balance NUMERIC(14, 2),
     products_number SMALLINT,
     credit_card SMALLINT,
     active_member SMALLINT,
     estimated_salary NUMERIC(14, 2),
     churn SMALLINT
 );
