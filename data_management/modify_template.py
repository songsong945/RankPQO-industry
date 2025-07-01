import os
import json

def process_directory(subdir):
    meta_data_path = os.path.join(subdir, "meta_data.json")
    
    # 确保 meta_data.json 和 parameter_new.json 文件存在
    if os.path.isfile(meta_data_path):
        print(f"Processing: {meta_data_path}")

        # 打开 meta_data.json 文件并加载其内容
        with open(meta_data_path, 'r') as f:
            data = json.load(f)
        
        # 检查是否包含 template 字段
        if 'template' in data:
            template = data['template']
            
            # 去掉 template 中的 'hd_income_band_sk = ib_income_band_sk AND'
            # modified_template = template.replace("hd_income_band_sk = ib_income_band_sk AND ", "")
            # modified_template = template.replace("%s0", "80")
            # modified_template = template.replace("%s8", "88")
            # modified_template = template.replace("2%s", "28")
            # modified_template = template.replace("AND %s", "AND 8")
            # modified_template = template.replace(" AND hd_income_band_sk = ib_income_band_sk;", ";")
            # modified_template = template.replace(", MIN(customer.c_income_band_sk) AS customer_c_income_band_sk", "")
            modified_template = template.replace("0.%s", "0.8")
    
            
            # 更新 template 字段
            data['template'] = modified_template
            
            # 将修改后的数据写回原文件
            with open(meta_data_path, 'w') as f:
                json.dump(data, f, indent=4)
        
        print(f"Processed: {meta_data_path}")
    else:
        print(f"Files not found in {subdir}")

def check_data(path):
    all_folders = []
    for subdir, _, files in os.walk(path):
        if ("meta_data.json" in files and "parameter_new.json" in files):
            with open(os.path.join(subdir, "parameter_new.json"), 'r') as f:
                parameter = json.load(f)
            
            if len(list(parameter.values())[0]) <= 1:
                meta_path = os.path.join(subdir, "meta_data.json")
                backup_path = os.path.join(subdir, "meta_data_bak.json")
                os.rename(meta_path, backup_path)
                print(f"Renamed: {meta_path} -> {backup_path}")

# 示例使用
# subdir = '../training_data/DSB_150/'  # 修改为目标文件夹的路径
# for x in os.walk(subdir):
#     process_directory(x[0])

subdir = '../training_data/JOB_330_2/' 
check_data(subdir)
