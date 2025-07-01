# Parametric Query Optimization in Industry

This project deploys **RankPQO** to **OceanBase** for parametric query optimization (PQO). It includes data collection, model training, plan selection, and online evaluation using the IMDB workload.

## 🚀 Workflow

1. Install OceanBase and load IMDB
2. Collect training data
3. Train the model
4. Select plans to cache
5. Evaluate online performance

---

## 1. Install OceanBase and Load IMDB

Follow the official guide: [OceanBase GitHub](https://github.com/oceanbase/oceanbase)

---

## 2. Collect Training Data

```bash
cd data_management/
python training_data_collect_time_ob.py
```

This script samples and times queries executed on OceanBase to prepare training data.

---

## 3. Train the Model

```bash
cd ..
python -m train.offline_training \
  --function hierarchical_cluster_train \
  --training_data ./training_data/JOB_330/ \
  --model_path ./checkpoints/JOB330_hierarchical_300_30/ \
  --device cuda \
  --epochs 10 \
  --epoch_step 1 \
  --cluster 30 \
  --data_size 300 \
  2>&1 | tee JOB330_hierarchical_300_30.log
```

---

## 4. Select Plans to Cache

```bash
python offline_plan_selection.py \
  --function candidate_selection_by_cluster \
  --training_data ./training_data/JOB_330/ \
  --model_path ./checkpoints/JOB330_hierarchical_300_30 \
  --device cuda \
  --k 30 \
  --output selected_plans_30.json \
  2>&1 | tee JOB330_hierarchical_plan_selection_30.log
```

---

## 5. Online Evaluation

```bash
python online_testing.py \
  --function eval_finetune_lazy \
  --training_data ./training_data/JOB_330/ \
  --model_path ./checkpoints/JOB330_hierarchical_300_30 \
  --device cuda \
  --group_dir ./data_management/parameters_330_test \
  --group_count 10 \
  --input_file selected_plans_30.json \
  --output_dir ./eval_output_lazy_300_30 \
  --max_plan_num 30 \
  --tau_max 2 \
  --max_steps 10 \
  2>&1 | tee eval_output_lazy_300_30.log
```
