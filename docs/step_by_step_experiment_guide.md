# MedicalGPT 完整实验操作手册：从零到部署的每一步

本文档是最详细的实操指南，适合你已经：
- 读完了理论
- 配好了服务器环境
- 准备开始第一轮完整实验

目标：手把手带你完成 `SFT → DPO → 合并 → 推理 → API 部署` 全流程。

---

## 0. 开始前的准备清单

在开始实验前，确认以下条件已满足：

### 0.1 硬件与环境
- [ ] 双 4090 或同等 GPU（显存 ≥ 24GB × 2）
- [ ] 内存 ≥ 100GB
- [ ] 磁盘空间 ≥ 200GB（模型 + 数据 + 输出）
- [ ] CUDA 11.8+ 或 12.1+
- [ ] Python 3.10 或 3.11

### 0.2 项目与依赖
- [ ] 项目已克隆到服务器
- [ ] 虚拟环境已创建并激活
- [ ] `requirements.txt` 已安装
- [ ] `torch.cuda.is_available()` 返回 `True`

### 0.3 数据与模型
- [ ] 基座模型已下载（推荐 `Qwen/Qwen2.5-7B-Instruct`）
- [ ] SFT 数据已准备（ShareGPT 格式）
- [ ] DPO 数据已准备（偏好对格式）

如果以上任何一项未完成，请先参考：
- [server_setup_guide.md](./server_setup_guide.md) - 环境配置
- [practice_roadmap_dual4090.md](./practice_roadmap_dual4090.md) - 路线规划

---

## 1. 第一步：数据准备与验证

### 1.1 检查 SFT 数据格式

SFT 训练需要 ShareGPT 格式的 jsonl 文件，每行一个 JSON 对象：

```json
{
  "conversations": [
    {"from": "human", "value": "什么是糖尿病？"},
    {"from": "gpt", "value": "糖尿病是一种代谢性疾病..."}
  ]
}
```

**验证命令**：
```bash
python validate_jsonl.py --file_path data/finetune/sharegpt_zh_1K_format.jsonl
```

**预期输出**：
```
✓ 文件格式正确
✓ 共 1000 条数据
✓ 所有对话包含 conversations 字段
```

### 1.2 检查 DPO 数据格式

DPO 训练需要偏好对数据，包含 5 个字段：

```json
{
  "system": "你是一个医疗助手",
  "history": [],
  "question": "如何预防感冒？",
  "response_chosen": "保持良好卫生习惯，勤洗手...",
  "response_rejected": "多喝热水就行了"
}
```

**验证命令**：
```bash
python - <<'PY'
import json
path = 'data/reward/dpo_zh_500.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())
    required = ['system', 'history', 'question', 'response_chosen', 'response_rejected']
    missing = [k for k in required if k not in sample]
    if missing:
        print(f"❌ 缺少字段: {missing}")
    else:
        print("✓ DPO 数据格式正确")
        print(f"✓ 字段: {list(sample.keys())}")
PY
```

### 1.3 准备小样本测试集

第一次训练建议先用小数据集验证流程：

```bash
# 从完整数据中提取前 100 条
head -100 data/finetune/sharegpt_zh_1K_format.jsonl > data/finetune/test_100.jsonl
head -50 data/reward/dpo_zh_500.jsonl > data/reward/test_50.jsonl
```

---

## 2. 第二步：SFT 训练（监督微调）

### 2.1 理解 SFT 的作用

SFT 让模型学会：
- 理解你的领域知识（医疗、法律等）
- 遵循特定的对话风格
- 回答领域内的专业问题

这是所有后续训练的基础。

### 2.2 SFT 训练命令（双 4090 推荐配置）

创建训练脚本 `my_run_sft.sh`：

```bash
#!/bin/bash

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

### 2.3 启动训练

```bash
# 赋予执行权限
chmod +x my_run_sft.sh

# 后台运行（推荐）
nohup bash my_run_sft.sh > logs/sft_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看日志
tail -f logs/sft_*.log
```

### 2.4 监控训练进度

**查看 GPU 使用情况**：
```bash
watch -n 1 nvidia-smi
```

**查看训练日志关键信息**：
```bash
grep -E "loss|epoch|step" logs/sft_*.log | tail -20
```

**使用 TensorBoard**：
```bash
tensorboard --logdir outputs-sft-qwen2.5-7b --port 6006
# 本地浏览器访问: http://<server_ip>:6006
```

### 2.5 判断训练是否正常

**正常信号**：
- ✓ Loss 逐步下降
- ✓ GPU 利用率 > 80%
- ✓ 每 step 耗时稳定（约 2-5 秒）
- ✓ 定期保存 checkpoint

**异常信号**：
- ❌ Loss 为 NaN 或 Inf → 学习率过大，降低到 1e-5
- ❌ OOM 错误 → 减小 `per_device_train_batch_size` 或 `model_max_length`
- ❌ GPU 利用率 < 50% → 增大 `gradient_accumulation_steps`

### 2.6 训练完成后的输出

训练完成后，`outputs-sft-qwen2.5-7b/` 目录包含：
- `adapter_config.json` - LoRA 配置
- `adapter_model.safetensors` - LoRA 权重
- `checkpoint-*/` - 训练检查点
- `trainer_state.json` - 训练状态

**重要**：这些文件是 LoRA adapter，不是完整模型，需要配合基座模型使用。

---

## 3. 第三步：SFT 模型推理验证

### 3.1 直接加载 LoRA 推理

```bash
python inference.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-sft-qwen2.5-7b \
  --interactive
```

**交互测试**：
```
User: 什么是高血压？
Assistant: [模型回答]

User: 如何预防糖尿病？
Assistant: [模型回答]
```

### 3.2 合并 LoRA 为完整模型

```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-sft-qwen2.5-7b \
  --output_dir merged-sft-qwen2.5-7b
```

**合并后推理**：
```bash
python inference.py \
  --base_model merged-sft-qwen2.5-7b \
  --interactive
```

### 3.3 批量对比测试

创建测试问题文件 `test_questions.txt`：
```
什么是糖尿病？
如何预防感冒？
高血压的症状有哪些？
```

**对比基座模型和 SFT 模型**：
```bash
python compare_batch_models.py \
  --models /data/models/Qwen2.5-7B-Instruct merged-sft-qwen2.5-7b \
  --model_names base sft \
  --questions_file test_questions.txt \
  --output_file sft_comparison.md
```

查看对比结果：
```bash
cat sft_comparison.md
```

---

## 4. 第四步：DPO 训练（偏好优化）

### 4.1 理解 DPO 的作用

DPO 让模型学会：
- 区分好回答和坏回答
- 生成更符合人类偏好的内容
- 避免有害、不准确或低质量的输出

DPO 基于 SFT 模型进行，所以需要先完成 SFT。

### 4.2 DPO 训练命令

创建训练脚本 `my_run_dpo.sh`：

```bash
#!/bin/bash

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
  --output_dir outputs-dpo-qwen2.5-7b \
  --report_to tensorboard \
  --cache_dir ./cache
```

### 4.3 启动 DPO 训练

```bash
chmod +x my_run_dpo.sh
nohup bash my_run_dpo.sh > logs/dpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/dpo_*.log
```

### 4.4 DPO 训练监控

**关键指标**：
- `loss` - 总损失，应逐步下降
- `rewards/chosen` - chosen 回答的奖励，应上升
- `rewards/rejected` - rejected 回答的奖励，应下降
- `rewards/margins` - 两者差距，应扩大

**查看指标**：
```bash
grep -E "loss|rewards" logs/dpo_*.log | tail -30
```

---

## 5. 第五步：DPO 模型验证与对比

### 5.1 合并 DPO LoRA

```bash
python merge_peft_adapter.py \
  --base_model merged-sft-qwen2.5-7b \
  --lora_model outputs-dpo-qwen2.5-7b \
  --output_dir merged-dpo-qwen2.5-7b
```

### 5.2 三模型对比测试

```bash
python compare_batch_models.py \
  --models /data/models/Qwen2.5-7B-Instruct merged-sft-qwen2.5-7b merged-dpo-qwen2.5-7b \
  --model_names base sft dpo \
  --questions_file test_questions.txt \
  --output_file full_comparison.md
```

### 5.3 分析对比结果

查看 `full_comparison.md`，重点关注：
- **Base → SFT**：是否学会了领域知识？
- **SFT → DPO**：回答是否更安全、更准确、更符合偏好？

---

## 6. 第六步：部署为 API 服务

### 6.1 启动 OpenAI 兼容 API

```bash
python openai_api.py \
  --model_name_or_path merged-dpo-qwen2.5-7b \
  --template_name qwen \
  --port 8000
```

**后台运行**：
```bash
nohup python openai_api.py \
  --model_name_or_path merged-dpo-qwen2.5-7b \
  --template_name qwen \
  --port 8000 \
  > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 6.2 测试 API

**使用 curl**：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "merged-dpo-qwen2.5-7b",
    "messages": [
      {"role": "user", "content": "什么是高血压？"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**使用 Python**：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="merged-dpo-qwen2.5-7b",
    messages=[
        {"role": "user", "content": "什么是糖尿病？"}
    ]
)

print(response.choices[0].message.content)
```

---

## 7. 完整流程时间估算

基于双 4090 + Qwen2.5-7B + 1K 样本：

| 步骤 | 预计时间 |
|------|---------|
| 数据准备与验证 | 30 分钟 |
| SFT 训练（1 epoch） | 2-4 小时 |
| SFT 推理验证 | 15 分钟 |
| DPO 训练（1000 steps） | 1-2 小时 |
| DPO 推理验证 | 15 分钟 |
| API 部署与测试 | 15 分钟 |
| **总计** | **5-8 小时** |

---

## 8. 常见问题与解决方案

### 8.1 OOM（显存不足）

**症状**：`CUDA out of memory`

**解决方案**：
1. 减小 `per_device_train_batch_size` 到 1
2. 减小 `model_max_length` 到 2048
3. 启用 `gradient_checkpointing`
4. 使用 `load_in_4bit=True`

### 8.2 训练速度慢

**症状**：每 step 超过 10 秒

**解决方案**：
1. 检查 GPU 利用率：`nvidia-smi`
2. 增大 `gradient_accumulation_steps`
3. 减少 `preprocessing_num_workers`
4. 确保使用 `bf16` 而非 `fp32`

### 8.3 Loss 不下降

**症状**：训练 100 steps 后 loss 仍然很高

**解决方案**：
1. 检查数据格式是否正确
2. 降低学习率到 1e-5
3. 增加 `warmup_ratio` 到 0.1
4. 检查是否有数据泄露（train/eval 重复）

### 8.4 推理结果不理想

**症状**：模型回答质量差

**解决方案**：
1. 检查是否加载了正确的模型
2. 增加训练数据量
3. 调整 `temperature` 和 `top_p`
4. 检查 prompt template 是否匹配

---

## 9. 实验记录模板

建议每次实验记录以下信息：

```markdown
## 实验 ID: exp_20260327_001

### 配置
- 基座模型: Qwen/Qwen2.5-7B-Instruct
- 训练类型: SFT
- 数据集: sharegpt_zh_1K_format.jsonl (1000 条)
- LoRA rank: 16
- 学习率: 2e-5
- Batch size: 1 × 16 (accumulation)
- Epochs: 1

### 结果
- 最终 loss: 0.85
- 训练时间: 3.2 小时
- 显存占用: 22GB × 2

### 评测
- Base vs SFT: SFT 明显更专业
- 问题: 部分回答过长

### 下一步
- 尝试增加 max_target_length 限制
```

---

## 10. 进阶优化方向

完成基础流程后，可以尝试：

1. **数据优化**
   - 增加高质量数据
   - 数据清洗与去重
   - 多轮对话数据

2. **训练优化**
   - 调整 LoRA rank（8/16/32/64）
   - 尝试不同学习率
   - 多 epoch 训练

3. **评测优化**
   - 建立标准测试集
   - 自动化评测脚本
   - 人工评测流程

4. **部署优化**
   - 模型量化（INT8/INT4）
   - 推理加速（vLLM）
   - 多模型负载均衡

---

## 11. 最后的建议

1. **第一次实验用小数据**：100 条 SFT + 50 条 DPO，快速验证流程
2. **记录每次实验**：参数、结果、问题，方便复现和对比
3. **先跑通再优化**：不要一开始就追求最优参数
4. **定期备份**：模型、日志、配置文件
5. **善用对比工具**：`compare_batch_models.py` 是你的好朋友

---

**祝实验顺利！如有问题，参考 [FAQ.md](./FAQ.md) 或提 Issue。**
