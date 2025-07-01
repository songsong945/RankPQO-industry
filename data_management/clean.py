import os

# 设置根目录
root_dir = '../training_data/JOB_330/'

# 保留的文件名
keep_files = {"meta_data.json", "hybrid_plans.json", "parameter_new.json"}

# 遍历每个子文件夹
for job_folder in os.listdir(root_dir):
    job_path = os.path.join(root_dir, job_folder)
    
    # 确保是文件夹
    if os.path.isdir(job_path):
        for file in os.listdir(job_path):
            file_path = os.path.join(job_path, file)
            # 删除不在保留名单中的文件
            if file not in keep_files:
                print(f"Deleting: {file_path}")
                os.remove(file_path)
