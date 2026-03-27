# MedicalGPT 实践路线图（双 4090 / 100G+ 内存）

这份文档面向这样一种场景：

- 你已经学完了大模型训练相关理论
- 你希望基于本仓库开始真正的实操
- 你有一台双 4090 服务器，内存 100G 以上

这类机器已经足够支撑本项目中最有价值的一条主线：

`SFT -> DPO -> Merge -> Inference/API`

不建议一开始就从 `PT` 或 `PPO` 入手。原因不是你的机器不够，而是它们对数据质量、实验设计和调参经验要求更高，作为第一条实践主线性价比不高。

## 1. 先说结论：推荐学习顺序

按优先级，推荐你这样学：

1. 读懂数据格式
2. 跑通 SFT 最小闭环
3. 学会做推理评测和模型对比
4. 跑 DPO 偏好优化
5. 合并 LoRA 并部署成可调用服务
6. 再回头看 PT、RM、PPO、GRPO

一句话概括：

先把“训练一个可用模型”做扎实，再去碰“训练一个更强但更复杂的模型”。

## 2. 为什么这条路线最适合你

你的硬件条件决定了你不是只能跑 Demo，而是可以做完整实验。但完整实验不等于一上来全都做。

本仓库里从学习收益和实操性来看，最应该优先吃透的是：

- `supervised_finetuning.py`
- `dpo_training.py`
- `merge_peft_adapter.py`
- `inference.py`
- `openai_api.py`

原因如下：

- `SFT` 是所有后续对齐训练的基础
- `DPO` 比 `RM + PPO` 链路更短，训练和排障成本更低
- `merge + inference + API` 能把“训练工程”闭环到“可用产品”
- 双 4090 跑 7B 级别的 `QLoRA + bf16` 非常合适

## 3. 仓库里你最该先看的文件

先看这几份文档和脚本：

- [docs/datasets.md](/Users/bilibili/PyCharmMiscProject/MedicalGPT/docs/datasets.md)
- [docs/scripts_overview_dual4090.md](/Users/bilibili/PyCharmMiscProject/MedicalGPT/docs/scripts_overview_dual4090.md)
- [docs/dpo_full_pipeline_server.md](/Users/bilibili/PyCharmMiscProject/MedicalGPT/docs/dpo_full_pipeline_server.md)
- [run_sft.sh](/Users/bilibili/PyCharmMiscProject/MedicalGPT/run_sft.sh)
- [run_dpo.sh](/Users/bilibili/PyCharmMiscProject/MedicalGPT/run_dpo.sh)
- [supervised_finetuning.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/supervised_finetuning.py)
- [dpo_training.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/dpo_training.py)
- [merge_peft_adapter.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/merge_peft_adapter.py)
- [inference.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/inference.py)
- [openai_api.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/openai_api.py)

如果你时间有限，优先顺序就是：

`datasets -> run_sft -> run_dpo -> inference -> api`

## 4. 第一阶段：先跑通 SFT 最小闭环

### 4.1 阶段目标

这一阶段的目标不是“训出最强模型”，而是先把完整链路跑明白：

- 数据怎么组织
- 参数怎么传进训练脚本
- 训练输出保存在哪里
- LoRA 和 merged model 的关系是什么
- 推理时加载什么模型

### 4.2 你需要做什么

1. 检查 SFT 数据格式  
   用 ShareGPT 格式 jsonl 数据，先用脚本校验：

```bash
python validate_jsonl.py --file_path data/finetune/sharegpt_zh_1K_format.jsonl
```

2. 直接跑项目默认 SFT 脚本  
   先别急着改一堆参数，先跑通：

```bash
sh run_sft.sh
```

3. 训练完成后做推理验证

```bash
python inference.py --base_model <merged_or_base_model_path> --interactive
```

### 4.3 这一步你必须搞明白的事

- `run_sft.sh` 只是入口，真正逻辑在 `supervised_finetuning.py`
- `--use_peft True` 表示训练 LoRA adapter，不是训练完整模型
- 输出目录里的内容不等于“可直接部署完整模型”
- LoRA 模型和 base model 通常需要一起使用，或者先 merge 再部署

## 5. 第二阶段：把 SFT 跑成正式实验

当你已经跑通最小闭环后，不要马上冲 DPO，先把 SFT 做成一轮像样的实验。

### 5.1 推荐配置

对于双 4090，推荐你把主要精力放在 7B 模型上，例如：

- `Qwen/Qwen2.5-7B-Instruct`

推荐训练方式：

- `QLoRA`
- `load_in_4bit=True`
- `bf16`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`

典型命令模板：

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
  --output_dir outputs-sft-qwen2.5-7b \
  --torch_dtype bfloat16 --bf16 \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --cache_dir ./cache
```

### 5.2 你应该重点观察什么

- train loss 是否稳定下降
- eval loss 是否同步变化
- 模型是否出现明显复读、答非所问、格式崩坏
- 长输入下是否出现截断问题
- 同一类医疗问答是否比 base model 更稳

### 5.3 这一步最容易犯的错误

- 一开始就上全量大数据，结果排错成本很高
- 一次改太多参数，最后不知道效果变化来自哪里
- 只看 loss，不看真实回答
- 训练完不做固定样本对比

## 6. 第三阶段：学会做模型对比

很多人会跑训练，但不会做实验。真正能把实践做扎实的人，都会建立自己的评测习惯。

推荐你固定一批 prompt，持续对比：

- base model
- SFT model
- DPO model

可以直接使用：

- [compare_batch_models.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/compare_batch_models.py)

建议你自己准备一份 `prompts.txt`，内容覆盖：

- 常规医疗问答
- 多轮追问
- 含歧义的用户表述
- 需要拒答或谨慎回答的问题
- 需要结构化输出的问题

这一步的目标是让你形成判断：

- 是数据问题
- 是模板问题
- 是训练问题
- 还是推理参数问题

## 7. 第四阶段：在 SFT 基础上做 DPO

这是你这台机器最值得做的第二个重点阶段。

### 7.1 为什么推荐 DPO，而不是先 PPO

因为：

- DPO 不需要先单独训练 RM 再走 PPO
- 训练链路更短
- 实验变量更少
- 更适合先建立偏好对齐直觉

### 7.2 正确顺序

推荐顺序是：

1. 先训练 SFT
2. merge SFT LoRA
3. 用 merge 后的 SFT 模型作为 DPO 基座
4. 再训练 DPO
5. 再 merge DPO LoRA

### 7.3 DPO 命令模板

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

### 7.4 DPO 数据你要重点关注什么

- `question` 是否清晰
- `chosen` 和 `rejected` 是否拉开差异
- 偏好标注是否稳定
- 是否混入明显低质量或冲突样本

偏好数据质量通常比数量更重要。

## 8. 第五阶段：Merge、推理、服务化

训练不是结束。模型只有被稳定加载、调用、验证，实践才算闭环。

### 8.1 Merge LoRA

```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-sft-qwen2.5-7b \
  --output_dir merged-sft-qwen2.5-7b
```

DPO 后同理：

```bash
python merge_peft_adapter.py \
  --base_model merged-sft-qwen2.5-7b \
  --lora_model outputs-dpo-qwen2.5-7b \
  --output_dir merged-dpo-qwen2.5-7b
```

### 8.2 本地推理验证

```bash
python inference.py --base_model merged-dpo-qwen2.5-7b --interactive
```

### 8.3 OpenAI 兼容 API

```bash
CUDA_VISIBLE_DEVICES=0 python openai_api.py \
  --base_model merged-dpo-qwen2.5-7b \
  --template_name qwen \
  --host 0.0.0.0 \
  --port 8000
```

这一步很重要，因为它会迫使你思考：

- 模型如何被外部系统接入
- 推理参数如何设置
- 服务如何做成稳定接口

## 9. 哪些方向先不要优先投入

### 9.1 PT（继续预训练）

对应：

- [pretraining.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/pretraining.py)

不建议第一时间投入的原因：

- 对语料质量要求高
- 数据清洗难度大
- 训练成本更高
- 很容易出现“训了很久但效果不明显”

PT 适合在以下情况下再考虑：

- 你已经有较高质量的大规模医疗原始语料
- 你明确知道模型缺的是领域知识，而不是指令跟随能力
- 你已经跑通 SFT 和 DPO

### 9.2 RM + PPO

对应：

- [reward_modeling.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/reward_modeling.py)
- [ppo_training.py](/Users/bilibili/PyCharmMiscProject/MedicalGPT/ppo_training.py)

不建议一开始就做的原因：

- 链路长
- 变量多
- 排错复杂
- 更适合有完整偏好数据和实验经验之后再做

### 9.3 GRPO / ORPO

它们值得学，但不应该抢在 `SFT + DPO` 前面。

建议顺序始终是：

`SFT -> DPO -> RM/PPO 或 ORPO/GRPO`

## 10. 实操时的建议原则

### 10.1 每次只改一个变量

不要同时改：

- 模型
- 数据
- 学习率
- 最大长度
- batch size
- LoRA rank

否则你无法判断变化来源。

### 10.2 先小样本调通，再全量训练

例如先用：

- `max_train_samples=1000`
- `max_eval_samples=50`

调通链路，再去掉限制。

### 10.3 一定要保留固定评测集

训练效果不是看“感觉”，要看同一组 prompt 的前后对比。

### 10.4 训练日志不是全部

loss 很重要，但不是唯一指标。医疗场景尤其要看：

- 是否过度自信
- 是否胡编药物和诊断
- 是否缺少风险提示
- 是否过度拒答

## 11. 推荐的 7 天实践安排

### Day 1

- 读 `docs/datasets.md`
- 跑 `validate_jsonl.py`
- 读 `run_sft.sh`
- 理解 SFT 输入输出结构

目标：搞清楚数据格式和训练入口。

### Day 2

- 用默认小样本配置跑通一轮 SFT
- 看输出目录
- 用 `inference.py` 交互验证

目标：把第一条训练链路跑通。

### Day 3

- 准备一批自己的测试 prompt
- 对比 base 和 sft
- 记录典型好例和坏例

目标：建立实验观察能力。

### Day 4

- 正式配置一轮双 4090 的 7B SFT
- 跑 TensorBoard
- 看 loss、看实际输出

目标：完成第一轮像样的 SFT 实验。

### Day 5

- merge SFT LoRA
- 准备 DPO 数据
- 检查 chosen/rejected 质量

目标：为 DPO 做高质量前置准备。

### Day 6

- 启动 DPO 训练
- 跟踪日志
- 对比 SFT 和 DPO 输出差异

目标：理解偏好对齐到底带来什么变化。

### Day 7

- merge DPO LoRA
- 跑 `openai_api.py`
- 用外部脚本或接口调用

目标：把训练结果变成可调用服务。

## 12. 你现阶段最应该追求什么

不是追求立刻做出最强模型，而是追求下面这几件事：

- 能独立准备数据
- 能独立跑通训练
- 能独立判断训练结果
- 能独立做 base/sft/dpo 对比
- 能把模型部署成一个可调用接口

只要这几件事做成了，你就已经从“学了理论”进入“能做项目”。

## 13. 最终建议

对于双 4090 + 100G+ 内存，最合理的路线是：

1. 先拿 `Qwen2.5-7B-Instruct` 做 `SFT`
2. 再在 SFT 基础上做 `DPO`
3. 再做 `merge + inference + API`
4. 把评测流程固定下来
5. 最后再考虑 `PT` 或 `RM/PPO`

如果你只想抓主线，就记住这一句：

**先把 `SFT -> DPO -> 部署` 做扎实，这就是你当前机器条件下最值得投入的实践路线。**
