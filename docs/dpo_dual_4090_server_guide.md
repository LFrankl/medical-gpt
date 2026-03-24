# MedicalGPT 双 4090 服务器 DPO 训练指南（现场下载模型版）

本文面向已经把 `MedicalGPT` 代码同步到服务器的场景，按“从 0 到可推理模型”给出完整终端流程。

## 1. 进入服务器与项目目录

```bash
ssh <user>@<server_ip>
cd /path/to/MedicalGPT
```

建议先进入 `tmux`，避免 SSH 断开导致训练中断：

```bash
tmux new -s dpo
```

## 2. 创建 Python 环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install bitsandbytes
```

## 3. 检查 GPU 与 PyTorch

```bash
nvidia-smi
python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpus=',torch.cuda.device_count())"
```

期望结果：
- `cuda=True`
- `gpus=2`（你的双 4090）

## 4. 现场下载 Qwen 模型到本地目录

先安装 Hugging Face CLI：

```bash
pip install -U "huggingface_hub[cli]"
```

如需访问受限模型，先登录（公开模型可跳过）：

```bash
huggingface-cli login
```

下载模型到本地：

```bash
mkdir -p /data/models/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
```

检查模型文件：

```bash
ls -lh /data/models/Qwen2.5-7B-Instruct
```

## 5. 检查 DPO 数据格式

`dpo_training.py` 期望每条样本至少包含字段：
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

## 6. 启动 DPO 训练（双卡 4090）

下面命令使用 LoRA + QLoRA（4bit），更适合 2x4090。

```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --template_name qwen \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 1 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --learning_rate 5e-5 \
  --max_steps 1000 \
  --eval_steps 100 \
  --save_steps 100 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --gradient_checkpointing True \
  --device_map auto \
  --report_to tensorboard \
  --remove_unused_columns False \
  --output_dir outputs-dpo-qwen2.5-7b \
  --cache_dir ./cache
```

## 7. 后台运行与日志查看（推荐）

如果你不在 `tmux` 里，建议用 `nohup`：

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --template_name qwen \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --do_train --do_eval \
  --use_peft True --qlora True --load_in_4bit True \
  --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --per_device_eval_batch_size 1 \
  --max_source_length 1024 --max_target_length 512 \
  --learning_rate 5e-5 --max_steps 1000 --eval_steps 100 --save_steps 100 \
  --torch_dtype bfloat16 --bf16 True --fp16 False \
  --gradient_checkpointing True --device_map auto \
  --report_to tensorboard --remove_unused_columns False \
  --output_dir outputs-dpo-qwen2.5-7b --cache_dir ./cache' \
  > dpo_train.log 2>&1 &
```

查看日志：

```bash
tail -f dpo_train.log
```

## 8. 训练完成后合并 LoRA Adapter

```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-dpo-qwen2.5-7b \
  --output_dir merged-dpo-qwen2.5-7b
```

## 9. 推理验证

```bash
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

## 10. TensorBoard 可视化（可选）

```bash
tensorboard --logdir outputs-dpo-qwen2.5-7b --host 0.0.0.0 --port 6006
```

## 11. 常见问题排查

1. 显存不够（OOM）
- 降低 `--per_device_train_batch_size`（优先改为 `1`）
- 增大 `--gradient_accumulation_steps`
- 缩短 `--max_source_length` / `--max_target_length`

2. 下载模型失败
- 检查网络连通性与 Hugging Face 访问
- 需要权限的模型先执行 `huggingface-cli login`

3. 数据字段报错（如 `KeyError: system`）
- 检查数据是否包含第 5 节列出的 5 个字段

4. 中断后续训
- 在原命令后追加：
```bash
--resume_from_checkpoint outputs-dpo-qwen2.5-7b/checkpoint-<step>
```

## 12. 最简执行清单

```bash
source .venv/bin/activate
nvidia-smi
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py ... --model_name_or_path /data/models/Qwen2.5-7B-Instruct
python merge_peft_adapter.py --base_model /data/models/Qwen2.5-7B-Instruct --lora_model outputs-dpo-qwen2.5-7b --output_dir merged-dpo-qwen2.5-7b
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

