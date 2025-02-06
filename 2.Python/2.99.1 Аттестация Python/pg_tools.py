import psycopg2 as pg
from pathlib import Path
import pandas as pd

# https://bbengfort.github.io/2017/12/psycopg2-transactions/

def connect(env="DATABASE_URL"):
    url = os.getenv(env)
    if not url:
        raise ValueError("no database url specified")
    return pg.connect(url)


def execute_sql_file(conn, sql_file=None):
    if sql_file is None:
        raise ValueError("no sql_file specified")
    if not Path(sql_file).is_file():
        raise ValueError(f"sql_file '{sql_file}' not found")
    
    with open(sql_file, 'r') as f:
        sql = f.read()

    try:
        with conn.cursor() as curs:
            curs.execute(sql)
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e

def execute_sql_file_to_dataframe(conn, sql_file=None):
    if sql_file is None:
        raise ValueError("no sql_file specified")
    if not Path(sql_file).is_file():
        raise ValueError(f"sql_file '{sql_file}' not found")
    
    with open(sql_file, 'r') as f:
        sql = f.read()

    try:
        with conn.cursor() as curs:
            curs.execute(sql)
            df = pd.DataFrame(curs.fetchall(), columns=[desc[0] for desc in curs.description])
            return df
    except Exception as e:
        conn.rollback()
        raise e

    
# Параметры подключения к БД
pg_user = 'mluser'
pg_password = 'mlpassword'
pg_host = 'localhost'
pg_database = 'pythonatt'
pg_port = '5433'
#pg_connection_string = f'postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}'
pg_connection_string = f'postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}'    