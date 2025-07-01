import psycopg2
import time
import os

import configure


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def execute_sql(query, connection):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    start_time = time.time()
    cursor.execute(query)
    end_time = time.time()

    cursor.close()

    return end_time - start_time


def process_sql_files(directory_path):
    connection = connect_to_pg()
    for filename in os.listdir(directory_path):
        if filename.endswith(".sql") and 'a' in filename:
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                print(f"prcessing query {filename} ...")
                query = file.read()
                execution_time = execute_sql(query, connection)
                print(f"Executed query from {filename} in {execution_time:.4f} seconds")


# 请替换为你的实际目录路径
data = '../training_data/join-order-benchmark/'
process_sql_files(data)
