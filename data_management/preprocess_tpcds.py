import os
import re
import json


def process_sql_query(query, filename):
    # 清除注释和定义行
    query = re.sub(r'--.*', '', query)  # 删除单行注释
    query = re.sub(r'define.*?;', '', query, flags=re.DOTALL)  # 删除定义行

    # 使用正则表达式找到从 "select" 开始的查询部分
    sql_query = re.search(r'select.*?;', query, re.DOTALL)
    if sql_query:
        sql_query = sql_query.group()
    else:
        return None

    # 替换占位符为 '%s'
    modified_query, num_subs = re.subn(r"'?\[(.*?)\]'?", "%s", sql_query)

    predicates = []
    # 逐行分析查询语句，以提取谓词
    for line in sql_query.splitlines():
        # 提取二元运算符
        binary_matches = re.findall(r"(\w+)\s*([=<>])\s*'?\[(.*?)\]'?", line)
        for column, operator, placeholder in binary_matches:
            predicates.append({
                "alias": None,  # 示例别名逻辑
                "column": column,  # 提取的列名
                "operator": operator,  # 提取的操作符
                "data_type": "text",  # 默认类型，根据需要调整
                "preprocess_type": "embedding",  # 示例预处理方式
                "distinct_values": [],  # 示例值
                "max_len": 1  # 示例最大长度
            })

        # 提取 BETWEEN AND
        between_matches = re.findall(r"(\w+)\s+BETWEEN\s+'\[(.*?)\]'\s+AND\s+'?\[(.*?)\]'?", line, re.IGNORECASE)
        for column, start, end in between_matches:
            predicates.append({
                "alias": None,
                "column": column,
                "operator": "between",
                "data_type": "int",  # BETWEEN通常用于数值
                "preprocess_type": "embedding",  # 示例预处理方式
                "distinct_values": [],  # 示例值
                "max_len": 1  # 示例最大长度
            })

        # 提取 IN
        in_matches = re.findall(r"(\w+)\s+IN\s+\((.*?)\)", line, re.IGNORECASE)
        for column, values in in_matches:
            values = re.findall(r"'?\[(.*?)\]'?", values)
            for value in values:
                predicates.append({
                    "alias": None,
                    "column": column,
                    "operator": "in",
                    "data_type": "text",  # IN 可用于多种数据类型
                    "preprocess_type": "embedding",  # 示例预处理方式
                    "distinct_values": [],  # 示例值
                    "max_len": 1  # 示例最大长度
                })

    # 格式化提取的 SQL 查询以符合模板格式
    sql_template = modified_query.strip().replace("\n", "\n       ")
    template_id = filename.split('_')[0]

    return {
        "template_id": template_id,
        "template": sql_template,
        "predicates": predicates
    }


def process_directory(directory_path):

    for filename in os.listdir(directory_path):
        if filename != 'query091_spj.tpl':
            continue
        if filename.endswith('.tpl'):  # 仅处理以 .tpl 结尾的文件
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    query = file.read()
                    template = process_sql_query(query, filename)
                    if template:
                        print(template)
                        # 创建输出目录
                        output_dir = os.path.join('../training_data/TPCDS/', template['template_id'])
                        # if not os.path.exists(output_dir):
                        # os.makedirs(output_dir)
                        # 写入 meta_data.json 文件
                        with open(os.path.join(output_dir, 'meta_data.json'), 'w') as json_file:
                            json.dump(template, json_file, indent=4)


# Example usage
directory_path = '../training_data/tpc-ds/'
process_directory(directory_path)
