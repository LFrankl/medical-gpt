# MedicalGPT 脚本精简版（双 4090 服务器实战）

这份是上一个总览的“实战裁剪版”，只保留你大概率会用到的脚本。

## 1. 你最需要的 10 个脚本

| 优先级 | 脚本 | 干什么 |
|---|---|---|
| 1 | `run_sft.sh` | 跑 SFT（监督微调） |
| 2 | `run_dpo.sh` | 跑 DPO（偏好优化） |
| 3 | `supervised_finetuning.py` | SFT 主程序（`run_sft.sh` 实际调用它） |
| 4 | `dpo_training.py` | DPO 主程序（`run_dpo.sh` 实际调用它） |
| 5 | `merge_peft_adapter.py` | 把 LoRA adapter 合并回基座模型，方便部署推理 |
| 6 | `inference.py` | 交互/批量推理验证模型效果 |
| 7 | `openai_api.py` | 起 OpenAI 兼容 API 服务（程序对接最方便） |
| 8 | `vllm_deployment.sh` | 用 vLLM 启服务（高吞吐部署） |
| 9 | `convert_dataset.py` | 把 Alpaca/QA 数据转 ShareGPT 格式 |
| 10 | `validate_jsonl.py` | 训练前检查 jsonl 格式是否合法 |

## 2. 你的推荐主线（最短路径）

1. 准备/校验数据  
`validate_jsonl.py`（SFT 数据） + 确认 DPO 数据字段完整

2. 跑 SFT  
`run_sft.sh`（或直接改参数跑 `supervised_finetuning.py`）

3. 跑 DPO  
`run_dpo.sh`（或直接改参数跑 `dpo_training.py`）

4. 合并 LoRA  
`merge_peft_adapter.py`

5. 推理验证  
`inference.py`

6. 部署服务  
`openai_api.py` 或 `vllm_deployment.sh`

## 3. 双 4090 常用命令模板

### 3.1 SFT（两卡）
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --train_file_dir ./data/finetune \
  --validation_file_dir ./data/finetune \
  --template_name qwen \
  --do_train --do_eval \
  --use_peft True \
  --qlora True --load_in_4bit True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 --torch_dtype bfloat16 \
  --output_dir outputs-sft-qwen2.5-7b
```

### 3.2 DPO（两卡）
```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --template_name qwen \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --do_train --do_eval \
  --use_peft True \
  --qlora True --load_in_4bit True \
  --target_modules all \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True --fp16 False --torch_dtype bfloat16 \
  --max_source_length 1024 --max_target_length 512 \
  --output_dir outputs-dpo-qwen2.5-7b
```

### 3.3 合并 LoRA
```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-dpo-qwen2.5-7b \
  --output_dir merged-dpo-qwen2.5-7b
```

### 3.4 本地推理验证
```bash
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

### 3.5 OpenAI 兼容 API 服务
```bash
CUDA_VISIBLE_DEVICES=0 python openai_api.py \
  --base_model merged-dpo-qwen2.5-7b \
  --template_name qwen \
  --host 0.0.0.0 \
  --port 8000
```

## 4. 哪些脚本你现在可以先不看

- `run_pt.sh` / `pretraining.py`：只有要做领域继续预训练才需要  
- `run_rm.sh` + `run_ppo.sh`：走 RLHF 全链路时才需要  
- `run_orpo.sh` / `run_grpo.sh`：属于 DPO 的替代或扩展路线  
- `merge_tokenizers.py` / `build_domain_tokenizer.py`：只有扩词表时需要

## 5. 你当前场景的建议

- 先固定一条稳定主线：`SFT -> DPO -> merge -> inference`
- 每阶段只改少量参数（先不同时改模型、数据、长度、batch）
- 先跑通小样本，再放开全量，避免长时间训练后才发现数据格式问题

