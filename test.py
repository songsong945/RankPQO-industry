import os
import json
import random

# 假设这是提供的字典
data = {
    "22a": [504],
    "29a": [146,729],
    "30a": [310,599,720,805,902]
}

for key, values in data.items():
    json_path = f"./training_data/JOB/{key}/parameter_new.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        for value in values:
            parameter_key = f"parameter {value + 1}"
            if parameter_key in json_data:
                print(parameter_key)
                new_value = random.randint(0, value - 1)
                json_data[parameter_key] = json_data[f"parameter {new_value + 1}"]

        # for value in values:
        #     parameter_key = f"parameter {value + 1}"
        #     if parameter_key in json_data:
        #         print(parameter_key)
        #         new_values = random_integers = [random.randint(0, 999) for _ in range(30)]
        #         for new_value in new_values:
        #             json_data[f"parameter {new_value + 1}"] = json_data[parameter_key]

        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)
    else:
        print(f"文件 {json_path} 不存在")
