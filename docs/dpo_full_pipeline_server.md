# MedicalGPT 服务器全流程指南：SFT 前置 + DPO 训练（双 4090）

本文用于你的实际场景：代码已在服务器，目标是在服务器上完整跑通 `SFT -> DPO -> 合并 -> 推理验证`。

## 0. 先说明：DPO 前置是什么

是的，通常建议先有一个可用的 SFT 模型，再做 DPO。

- 直接用官方 Instruct 模型做 DPO：可行，但对齐增益不一定稳定
- 先用你的领域数据做 SFT，再做 DPO：更符合“先学任务，再学偏好”的流程

本文按推荐路线写：先 SFT，再 DPO。

## 1. 环境准备

### 1.1 登录服务器并进入项目
```bash
ssh <user>@<server_ip>
cd /path/to/MedicalGPT
```

建议先开 `tmux`，防止 SSH 断开：
```bash
tmux new -s medgpt
```

### 1.2 创建 Python 环境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install bitsandbytes
pip install -U "huggingface_hub[cli]"
```

### 1.3 检查 GPU
```bash
nvidia-smi
python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpus=',torch.cuda.device_count())"
```

期望：`cuda=True` 且 `gpus=2`（双 4090）。

## 2. 模型下载（现场下载）

```bash
huggingface-cli login   # 如模型需要权限则登录；公开模型可跳过
mkdir -p /data/models/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
ls -lh /data/models/Qwen2.5-7B-Instruct
```

后续所有 `--model_name_or_path` 用本地路径，避免重复拉取。

## 3. 数据准备与检查

### 3.1 SFT 数据（ShareGPT 格式）
推荐先检查：
```bash
python validate_jsonl.py --file_path data/finetune/sharegpt_zh_1K_format.jsonl
```

如你是 Alpaca/QA 数据，先转格式：
```bash
python convert_dataset.py \
  --in_file <your_data>.json \
  --out_file <your_data>_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json
```

### 3.2 DPO 数据（偏好格式）
`dpo_training.py` 期望字段：
- `system`
- `history`
- `question`
- `response_chosen`
- `response_rejected`

快速检查：
```bash
python - <<'PY'
import json
p='data/reward/dpo_zh_500.jsonl'
x=json.loads(open(p,'r',encoding='utf-8').readline())
print(x.keys())
PY
```

## 4. 第一步：SFT 训练（前置）

这里给双 4090 的稳定参数模板（QLoRA）。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --train_file_dir ./data/finetune \
  --validation_file_dir ./data/finetune \
  --template_name qwen \
  --do_train --do_eval \
  --use_peft True \
  --qlora True --load_in_4bit True \
  --target_modules all \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --model_max_length 4096 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 0.05 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 5 \
  --preprocessing_num_workers 4 \
  --output_dir outputs-sft-qwen2.5-7b \
  --torch_dtype bfloat16 --bf16 \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --cache_dir ./cache
```

### 4.1 SFT 后台运行（可选）
```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py ...' > sft.log 2>&1 &
tail -f sft.log
```

## 5. 第二步：合并 SFT LoRA（给 DPO 当基座）

```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-sft-qwen2.5-7b \
  --output_dir merged-sft-qwen2.5-7b
```

说明：DPO 的 `--model_name_or_path` 建议指向 `merged-sft-qwen2.5-7b`，而不是原始基座。

## 6. 第三步：DPO 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path merged-sft-qwen2.5-7b \
  --template_name qwen \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --do_train --do_eval \
  --use_peft True \
  --qlora True --load_in_4bit True \
  --target_modules all \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --learning_rate 5e-5 \
  --max_steps 1000 \
  --eval_steps 100 \
  --save_steps 100 \
  --torch_dtype bfloat16 \
  --bf16 True --fp16 False \
  --device_map auto \
  --gradient_checkpointing True \
  --remove_unused_columns False \
  --report_to tensorboard \
  --output_dir outputs-dpo-qwen2.5-7b \
  --cache_dir ./cache
```

### 6.1 DPO 后台运行（可选）
```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py ...' > dpo.log 2>&1 &
tail -f dpo.log
```

## 7. 第四步：合并 DPO LoRA

```bash
python merge_peft_adapter.py \
  --base_model merged-sft-qwen2.5-7b \
  --lora_model outputs-dpo-qwen2.5-7b \
  --output_dir merged-dpo-qwen2.5-7b
```

## 8. 第五步：推理验证

```bash
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

如要 API 化：
```bash
CUDA_VISIBLE_DEVICES=0 python openai_api.py \
  --base_model merged-dpo-qwen2.5-7b \
  --template_name qwen \
  --host 0.0.0.0 \
  --port 8000
```

## 9. 监控与可视化

```bash
tensorboard --logdir outputs-sft-qwen2.5-7b --host 0.0.0.0 --port 6006
tensorboard --logdir outputs-dpo-qwen2.5-7b --host 0.0.0.0 --port 6007
```

## 10. 断点续训

### 10.1 SFT 续训
在原命令追加：
```bash
--resume_from_checkpoint outputs-sft-qwen2.5-7b/checkpoint-<step>
```

### 10.2 DPO 续训
在原命令追加：
```bash
--resume_from_checkpoint outputs-dpo-qwen2.5-7b/checkpoint-<step>
```

## 11. 常见问题

1. OOM（显存不足）
- `per_device_train_batch_size` 保持 `1`
- 增大 `gradient_accumulation_steps`
- 降低 `model_max_length` 或 `max_source_length/max_target_length`

2. 数据字段报错
- SFT 检查 ShareGPT 格式
- DPO 检查 5 个字段是否齐全

3. 模型下载失败
- 检查网络
- 需要权限则 `huggingface-cli login`

4. 训练很慢
- 先用小样本调通：`--max_train_samples 1000 --max_eval_samples 50`
- 跑通后再去掉样本上限

## 12. 最短执行清单（复制即用）

```bash
source .venv/bin/activate
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct

# 1) SFT
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py ... \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --output_dir outputs-sft-qwen2.5-7b

# 2) merge SFT
python merge_peft_adapter.py --base_model /data/models/Qwen2.5-7B-Instruct --lora_model outputs-sft-qwen2.5-7b --output_dir merged-sft-qwen2.5-7b

# 3) DPO
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py ... \
  --model_name_or_path merged-sft-qwen2.5-7b \
  --output_dir outputs-dpo-qwen2.5-7b

# 4) merge DPO
python merge_peft_adapter.py --base_model merged-sft-qwen2.5-7b --lora_model outputs-dpo-qwen2.5-7b --output_dir merged-dpo-qwen2.5-7b

# 5) inference
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

