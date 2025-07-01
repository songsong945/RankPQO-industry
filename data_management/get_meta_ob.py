import json
import pymysql
import re
from decimal import Decimal
import configure
import os

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def connect_to_oceanbase():
    """
    连接 OceanBase 数据库
    """
    connection = pymysql.connect(
        host="127.0.0.1",       # 数据库主机地址
        port=2881,             # 端口号
        user="root",        # 用户名（包含租户信息）
        password="Y9dU0FkKYxjwrH1MCBR2",            # 密码（请填写正确的密码）
        database="DSB",         # 目标数据库
        charset="utf8mb4",      # 字符集
        local_infile=True       # 允许加载本地文件
    )
    return connection

def add_statistics(meta_data_path):
    connection = connect_to_oceanbase()
    cursor = connection.cursor()

    # 读取 meta_data 文件
    with open(meta_data_path, 'r') as file:
        meta_data = json.load(file)

    column_to_table = {
        "ss_sales_price": "store_sales",
        "ss_wholesale_cost": "store_sales",
        "cs_wholesale_cost": "catalog_sales",
        "cs_list_price": "catalog_sales",
        "ws_wholesale_cost": "web_sales",
        "cd_marital_status": "customer_demographics",
        "cd_education_status": "customer_demographics",
        "cd_dep_count": "customer_demographics",
        "hd_dep_count": "household_demographics",
        "hd_buy_potential": "household_demographics",
        "hd_income_band_sk": "household_demographics",
        "i_category": "item",
        "i_manager_id": "item",
        "c_birth_month": "customer",
        "d_year": "date_dim",
        "d_moy": "date_dim",
        "s_state": "store",
        "ca_state": "customer_address",
        "ca_city": "customer_address",
        "ca_gmt_offset": "customer_address",
        "w_gmt_offset": "warehouse",
        "cc_class": "call_center",
        "sm_type": "ship_mode",
        "inv_quantity_on_hand": "inventory",
        "cs_quantity": "catalog_sales",
        "ib_lower_bound": "income_band",
        "ib_upper_bound": "income_band",
        "ws_sales_price": "web_sales",
        "ss_list_price": "store_sales",
        "ss_quantity": "store_sales"
    }

    # 使用配置文件中的表名映射
    # table_mapping = configure.mapping_tpcds

    # 遍历每个谓词，获取统计信息
    for predicate in meta_data["predicates"]:
        table_column = predicate["column"]
        if '.' in table_column:
            alias, column_name = table_column.split(".")
        else:
            column_name = table_column
        # 根据完整列名映射获取实际表名
        # print(table_column)
        # print(column_name)
        actual_table_name = column_to_table.get(column_name)

        # 对于 int 类型
        if predicate["data_type"] == "int":
            query = f"SELECT MIN({column_name}), MAX({column_name}), COUNT(DISTINCT {column_name}) FROM {actual_table_name}"
            cursor.execute(query)
            min_val, max_val, distinct_count = cursor.fetchone()
            predicate["min"] = min_val
            predicate["max"] = max_val
            predicate["max_len"] = distinct_count + 5

        # 对于 float 类型
        elif predicate["data_type"] == "float":
            query = f"SELECT AVG({column_name}), VARIANCE({column_name}) FROM {actual_table_name}"
            cursor.execute(query)
            mean, variance = cursor.fetchone()
            predicate["mean"] = mean
            predicate["variance"] = variance

        # 对于 text 类型
        elif predicate["data_type"] == "text":
            # 查询该列所有不重复的值
            query = f"SELECT DISTINCT {column_name} FROM {actual_table_name} LIMIT 5000"
            cursor.execute(query)
            distinct_rows = cursor.fetchall()
            
            distinct_values = [row[0] for row in distinct_rows]
            distinct_count = len(distinct_values)
            
            predicate["distinct_values"] = distinct_values
            predicate["max_len"] = distinct_count + 5

    # 关闭数据库连接
    cursor.close()
    connection.close()

    # 更新 meta_data.json 文件
    with open(meta_data_path, 'w') as file:
        json.dump(meta_data, file, indent=4, default=decimal_default)

def process_all_meta_data(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")
        if os.path.isfile(meta_data_path):
            print(f"Processing: {meta_data_path}")
            add_statistics(meta_data_path)

if __name__ == "__main__":
    meta_data_path = "./temp_1500"
    process_all_meta_data(meta_data_path)
