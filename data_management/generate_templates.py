import networkx as nx
import random
import os

random.seed(42)

def create_db_graph_dsb():
    G = nx.Graph()
    
    # 添加表（节点）及其列信息
    dsb_tables = {
        "call_center": ["cc_call_center_sk", "cc_call_center_id", "cc_rec_start_date", "cc_rec_end_date"],
        "catalog_page": ["cp_catalog_page_sk", "cp_catalog_page_id", "cp_start_date_sk", "cp_end_date_sk"],
        "catalog_returns": ["cr_returned_date_sk", "cr_returned_time_sk", "cr_item_sk", "cr_refunded_customer_sk"],
        "catalog_sales": ["cs_sold_date_sk", "cs_sold_time_sk", "cs_ship_date_sk", "cs_bill_customer_sk", "cs_item_sk", "cs_promo_sk", "cs_order_number"],
        "customer": ["c_customer_sk", "c_customer_id", "c_current_cdemo_sk", "c_current_hdemo_sk", "c_current_addr_sk", "c_income_band_sk"],
        "customer_address": ["ca_address_sk", "ca_address_id", "ca_state", "ca_country"],
        "customer_demographics": ["cd_demo_sk", "cd_gender", "cd_marital_status"],
        "date_dim": ["d_date_sk", "d_date_id", "d_year", "d_moy"],
        "dbgen_version": ["dv_version"],
        "household_demographics": ["hd_demo_sk", "hd_income_band_sk", "hd_buy_potential"],
        "income_band": ["ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"],
        "inventory": ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"],
        "item": ["i_item_sk", "i_item_id", "i_brand", "i_class"],
        "promotion": ["p_promo_sk", "p_promo_id", "p_start_date_sk", "p_end_date_sk"],
        "reason": ["r_reason_sk", "r_reason_id", "r_reason_desc"],
        "ship_mode": ["sm_ship_mode_sk", "sm_ship_mode_id", "sm_type"],
        "store": ["s_store_sk", "s_store_id", "s_state", "s_country"],
        "store_returns": ["sr_returned_date_sk", "sr_item_sk", "sr_customer_sk"],
        "store_sales": ["ss_sold_date_sk", "ss_sold_time_sk", "ss_customer_sk", "ss_item_sk", "ss_promo_sk", "ss_store_sk", "ss_quantity"],
        "time_dim": ["t_time_sk", "t_time_id", "t_hour", "t_minute"],
        "warehouse": ["w_warehouse_sk", "w_warehouse_id", "w_state"],
        "web_page": ["wp_web_page_sk", "wp_web_page_id", "wp_creation_date_sk"],
        "web_returns": ["wr_returned_date_sk", "wr_item_sk", "wr_refunded_customer_sk"],
        "web_sales": ["ws_sold_date_sk", "ws_sold_time_sk", "ws_ship_date_sk", "ws_bill_customer_sk", "ws_item_sk", "ws_web_page_sk", "ws_web_site_sk"],
        "web_site": ["web_site_sk", "web_site_id", "web_name", "web_country"]
    }

    
    for table, columns in dsb_tables.items():
        G.add_node(table, columns=columns)
    
    # 添加外键关系（边），同时记录连接条件
    edges = [
        ("store_sales", "customer", "ss_customer_sk = c_customer_sk"),
        ("store_sales", "item", "ss_item_sk = i_item_sk"),
        ("store_sales", "promotion", "ss_promo_sk = p_promo_sk"),
        ("store_sales", "store", "ss_store_sk = s_store_sk"),
        ("store_sales", "date_dim", "ss_sold_date_sk = d_date_sk"),
        ("store_sales", "time_dim", "ss_sold_time_sk = t_time_sk"),

        ("store_returns", "customer", "sr_customer_sk = c_customer_sk"),
        ("store_returns", "item", "sr_item_sk = i_item_sk"),
        ("store_returns", "date_dim", "sr_returned_date_sk = d_date_sk"),

        ("web_sales", "customer", "ws_bill_customer_sk = c_customer_sk"),
        ("web_sales", "item", "ws_item_sk = i_item_sk"),
        ("web_sales", "web_page", "ws_web_page_sk = wp_web_page_sk"),
        ("web_sales", "web_site", "ws_web_site_sk = web_site_sk"),
        ("web_sales", "date_dim", "ws_sold_date_sk = d_date_sk"),
        ("web_sales", "time_dim", "ws_sold_time_sk = t_time_sk"),

        ("web_returns", "customer", "wr_refunded_customer_sk = c_customer_sk"),
        ("web_returns", "item", "wr_item_sk = i_item_sk"),
        ("web_returns", "date_dim", "wr_returned_date_sk = d_date_sk"),

        ("catalog_sales", "customer", "cs_bill_customer_sk = c_customer_sk"),
        ("catalog_sales", "item", "cs_item_sk = i_item_sk"),
        ("catalog_sales", "promotion", "cs_promo_sk = p_promo_sk"),
        ("catalog_sales", "date_dim", "cs_sold_date_sk = d_date_sk"),
        ("catalog_sales", "time_dim", "cs_sold_time_sk = t_time_sk"),

        ("inventory", "item", "inv_item_sk = i_item_sk"),
        ("inventory", "warehouse", "inv_warehouse_sk = w_warehouse_sk"),
        ("inventory", "date_dim", "inv_date_sk = d_date_sk"),

        ("customer", "customer_address", "c_current_addr_sk = ca_address_sk"),
        ("customer", "customer_demographics", "c_current_cdemo_sk = cd_demo_sk"),
        ("customer", "household_demographics", "c_current_hdemo_sk = hd_demo_sk"),
        ("household_demographics", "income_band", "hd_income_band_sk = ib_income_band_sk")
    ]
    
    for table1, table2, condition in edges:
        G.add_edge(table1, table2, condition=condition)
    
    return G

def create_db_graph():
    G = nx.Graph()
    
    # 添加表（节点）及其列信息
    tables = {
        "aka_name": ["id", "person_id", "name"],
        "aka_title": ["id", "movie_id", "title", "kind_id"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "role_id", "note", "nr_order"],
        "char_name": ["id", "name"],
        "comp_cast_type": ["id", "kind"],
        "company_name": ["id", "name", "country_code"],
        "company_type": ["id", "kind"],
        "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
        "info_type": ["id", "info"],
        "keyword": ["id", "keyword"],
        "kind_type": ["id", "kind"],
        "link_type": ["id", "link"],
        "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
        "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
        "name": ["id", "name", "gender"],
        "person_info": ["id", "person_id", "info_type_id", "info"],
        "role_type": ["id", "role"],
        "title": ["id", "title", "kind_id", "production_year", "episode_nr"]
    }
    
    for table, columns in tables.items():
        G.add_node(table, columns=columns)
    
    # 添加外键关系（边），同时记录连接条件
    edges = [
        ("aka_name", "name", "aka_name.person_id = name.id"),
        ("aka_title", "title", "aka_title.movie_id = title.id"),
        ("cast_info", "name", "cast_info.person_id = name.id"),
        ("cast_info", "title", "cast_info.movie_id = title.id"),
        ("cast_info", "char_name", "cast_info.person_role_id = char_name.id"),
        ("cast_info", "role_type", "cast_info.role_id = role_type.id"),
        ("company_name", "movie_companies", "company_name.id = movie_companies.company_id"),
        ("company_type", "movie_companies", "company_type.id = movie_companies.company_type_id"),
        ("complete_cast", "title", "complete_cast.movie_id = title.id"),
        ("info_type", "movie_info", "info_type.id = movie_info.info_type_id"),
        ("info_type", "movie_info_idx", "info_type.id = movie_info_idx.info_type_id"),
        ("info_type", "person_info", "info_type.id = person_info.info_type_id"),
        ("keyword", "movie_keyword", "keyword.id = movie_keyword.keyword_id"),
        ("kind_type", "title", "kind_type.id = title.kind_id"),
        ("kind_type", "aka_title", "kind_type.id = aka_title.kind_id"),
        ("link_type", "movie_link", "link_type.id = movie_link.link_type_id"),
        ("movie_companies", "title", "movie_companies.movie_id = title.id"),
        ("movie_info", "title", "movie_info.movie_id = title.id"),
        ("movie_info_idx", "title", "movie_info_idx.movie_id = title.id"),
        ("movie_keyword", "title", "movie_keyword.movie_id = title.id"),
        ("movie_link", "title", "movie_link.movie_id = title.id"),
        ("person_info", "name", "person_info.person_id = name.id"),
    ]

    
    for table1, table2, condition in edges:
        G.add_edge(table1, table2, condition=condition)
    
    return G

def generate_sql_template_dsb(G, min_tables=6, max_tables=13):
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

    tables = list(G.nodes)
    num = random.random()
    if num > 0.95:
        num_tables = random.randint(min_tables, max_tables)
    else:
        num_tables = random.randint(min_tables, 9)
    selected_tables = random.sample(tables, num_tables)
    
    # 确保选中的表是连通的
    subgraph = G.subgraph(selected_tables)
    while not nx.is_connected(subgraph):
        selected_tables = random.sample(tables, num_tables)
        subgraph = G.subgraph(selected_tables)
    
    # 生成 SELECT 语句（随机选择 3-12 列）
    select_columns = []
    num_select = random.randint(3, min(num_tables,12))
    for table in random.sample(selected_tables, num_select):
        column = random.choice(G.nodes[table]['columns'])
        select_columns.append(f"MIN({table}.{column}) AS {table}_{column}")
    
    select_clause = "SELECT " + ", ".join(select_columns)
    from_clause = "FROM " + ", ".join(selected_tables)
    
    # # 预定义的 WHERE 条件集（完整提取自 15 个查询）
    predefined_conditions = [
        "ss_sales_price BETWEEN 100.00 AND 150.00",
        "ss_wholesale_cost BETWEEN 80 AND 100",
        "cs_wholesale_cost BETWEEN 69 AND 88",
        "cs_list_price BETWEEN 77 AND 106",
        "ws_wholesale_cost BETWEEN 80 AND 100",
        "cd_marital_status = 'D'",
        "cd_education_status = 'Primary'",
        "cd_dep_count BETWEEN 6 AND 8",
        "hd_dep_count = 3",
        "hd_buy_potential = '1001-5000'",
        "hd_income_band_sk BETWEEN 14 AND 20",
        "i_category = 'Jewelry'",
        "i_manager_id BETWEEN 28 AND 67",
        "c_birth_month = 4",
        "d_year = 2001",
        "d_moy = 8",
        "s_state = 'IL'",
        "ca_state IN ('IN', 'MT', 'GA')",
        "ca_city = 'Hopewell'",
        "ca_gmt_offset = -7",
        "w_gmt_offset = -5",
        "cc_class = 'small'",
        "sm_type = 'TWO DAY'",
        "ib_lower_bound >= 30000",
        "ib_upper_bound <= 80000",
        "ws_sales_price BETWEEN 100.00 AND 150.00",
        "ss_sales_price / ss_list_price BETWEEN 0.8 AND 1.0"
    ]

    
    where_clauses = [edge[2]['condition'] for edge in subgraph.edges(data=True)]
    num_conditions = random.randint(8, len(predefined_conditions)) 

    temp_conditions = random.sample(predefined_conditions, min(len(predefined_conditions), num_conditions))

    #print(selected_tables)
    for condition in temp_conditions:
        pre_con = condition.split(' ')[0]
        #print(pre_con)
        table = column_to_table.get(pre_con)
        if table in selected_tables:
            where_clauses.append(condition)

    where_clause = "WHERE " + " AND ".join(where_clauses)
    
    sql_query = f"{select_clause} {from_clause} {where_clause};"
    return sql_query

def generate_sql_template(G, min_tables=3, max_tables=16):
    tables = list(G.nodes)
    num = random.random()
    if num > 0.95:
        num_tables = random.randint(min_tables, max_tables)
    else:
        num_tables = random.randint(min_tables, 6)
    selected_tables = random.sample(tables, num_tables)
    
    # 确保选中的表是连通的
    subgraph = G.subgraph(selected_tables)
    while not nx.is_connected(subgraph):
        selected_tables = random.sample(tables, num_tables)
        subgraph = G.subgraph(selected_tables)
    
    # 生成 SELECT 语句（随机选择 1-3 列）
    select_columns = []
    num_select = random.randint(1, 3)
    for table in random.sample(selected_tables, num_select):
        column = random.choice(G.nodes[table]['columns'])
        select_columns.append(f"MIN({table}.{column}) AS {table}_{column}")
    
    select_clause = "SELECT " + ", ".join(select_columns)
    from_clause = "FROM " + ", ".join(selected_tables)
    
    # # 预定义的 WHERE 条件集（完整提取自 33 个查询）
    predefined_conditions = [
        "cast_info.note LIKE '%(voice)%'",
        "cast_info.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)')",
        "company_name.country_code = '[ru]'",
        "company_name.country_code != '[us]'",
        "company_name.name LIKE '%Film%'",
        "company_name.name NOT LIKE '%Warner%'",
        "company_type.kind = 'production companies'",
        "info_type.info = 'rating'",
        "keyword.keyword IN ('murder', 'violence', 'blood', 'gore', 'death')",
        "kind_type.kind IN ('movie', 'episode', 'series')",
        "link_type.link LIKE '%follow%'",
        "movie_companies.note LIKE '%(200%)%'",
        "movie_companies.note NOT LIKE '%(USA)%'",
        "movie_info.info LIKE 'USA:% 200%'",
        "movie_info.info IN ('Drama', 'Horror', 'Thriller', 'Action', 'Sci-Fi')",
        "movie_info.note LIKE '%internet%'",
        "movie_info_idx.info < '7.0'",
        "movie_info_idx.info > '8.0'",
        "person_info.note = 'Volker Boehm'",
        "title.production_year BETWEEN 1955 AND 2000",
        "title.production_year > 2005",
        "title.production_year < 2010",
        "title.episode_nr > 50",
        "title.episode_nr < 100",
        "role_type.role = 'actor'",
        "name.gender = 'm'",
        "name.name LIKE 'B%'",
        "name.name NOT LIKE '%Tim%'",
        "name.name_pcode_cf BETWEEN 'A' AND 'F'"
    ]
    
    where_clauses = [edge[2]['condition'] for edge in subgraph.edges(data=True)]
    num_conditions = random.randint(8, len(predefined_conditions)) 

    temp_conditions = random.sample(predefined_conditions, min(len(predefined_conditions), num_conditions))

    # print(selected_tables)
    for condition in temp_conditions:
        pre_con_table = condition.split('.')[0]
        # print(pre_con_table)
        if pre_con_table in selected_tables:
            where_clauses.append(condition)

    where_clause = "WHERE " + " AND ".join(where_clauses)
    
    sql_query = f"{select_clause} {from_clause} {where_clause};"
    return sql_query

# # 创建数据库图
# G = create_db_graph()

# output_dir = "./job_3300_2"
# os.makedirs(output_dir, exist_ok=True)

# # 生成 SQL 查询模板并写入文件
# for i in range(3300):
#     sql_query = generate_sql_template(G)
#     with open(f"{output_dir}/job_{i+1}.sql", "w") as f:
#         f.write(sql_query)


# 创建数据库图
G = create_db_graph_dsb()

output_dir = "./dsb_1500"
os.makedirs(output_dir, exist_ok=True)

# 生成 SQL 查询模板并写入文件
for i in range(1500):
    sql_query = generate_sql_template_dsb(G)
    with open(f"{output_dir}/dsb_{i+1}.sql", "w") as f:
        f.write(sql_query)
