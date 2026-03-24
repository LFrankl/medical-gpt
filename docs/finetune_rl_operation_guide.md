# MedicalGPT 微调 / 强化学习操作指南

这份指南不讲原理，重点讲这个仓库里每个环节该怎么做，适合你已经有理论基础、现在要自己做实验的人。

目标是帮你完成下面几类工作：

1. 跑通自己的 SFT 微调
2. 跑偏好对齐，包括 DPO / ORPO / RM + PPO
3. 自己造数据，并接进训练流水线
4. 跑 base model 和自己模型的对比实验
5. 把每一步的输入、输出、衔接关系理清楚

## 1. 先看结论：这个项目怎么用最合理

如果你的目标是先做出稳定结果，推荐主线不是全都跑，而是按下面顺序推进：

1. 选一个兼容性好的 base model
2. 准备自己的 SFT 数据
3. 跑 LoRA SFT
4. 合并 LoRA，得到可直接加载的模型
5. 准备偏好数据
6. 跑 DPO 或 ORPO
7. 再次合并 LoRA
8. 用 `inference.py` 做人工抽查和批量对比

如果你的目标是把 RLHF 全链路也走通，再走：

1. 跑 SFT
2. 合并 SFT LoRA
3. 准备偏好数据
4. 跑 RM
5. 合并 RM LoRA
6. 跑 PPO
7. 做和 base / SFT / DPO 的对比

## 2. 这个仓库里各阶段对应什么脚本

最常用的是这些：

- `run_sft.sh` -> `supervised_finetuning.py`
- `run_full_sft.sh` -> `supervised_finetuning.py`
- `run_rm.sh` -> `reward_modeling.py`
- `run_ppo.sh` -> `ppo_training.py`
- `run_dpo.sh` -> `dpo_training.py`
- `run_orpo.sh` -> `orpo_training.py`
- `run_grpo.sh` -> `grpo_training.py`
- `merge_peft_adapter.py` -> 把 LoRA adapter 合并回 base
- `inference.py` -> 本地交互 / 批量推理
- `convert_dataset.py` -> 把 Alpaca / QA 转成 ShareGPT
- `validate_jsonl.py` -> 校验 SFT jsonl 格式

你真正要熟的是 4 类输入输出关系：

- SFT 输入 ShareGPT 对话数据，输出 SFT adapter 或全参模型
- RM / DPO / ORPO 输入偏好数据，输出对齐后的 adapter
- PPO 输入合并后的 SFT 模型 + 合并后的 RM 模型 + 提示数据，输出 PPO 模型
- 推理评测阶段输入 base 或 merged model，输出生成结果

## 3. 推荐的实验目录组织

仓库现在自带的 `data/` 只是样例。你自己做实验时，建议单独整理，避免和样例混在一起。

建议目录：

```text
data/
  custom/
    sft/
      train.jsonl
      val.jsonl
    reward/
      train.jsonl
      val.jsonl
    ppo/
      train.jsonl
      val.jsonl
    eval/
      prompts.txt
      sft_eval.jsonl
      preference_eval.jsonl

outputs/
  sft/
  rm/
  dpo/
  orpo/
  ppo/
  merged/
  predictions/
```

这样做的好处：

- 每种训练的数据格式不同，不容易混
- 每个阶段的输出能单独管理
- 后面做对比实验时，不会搞不清楚哪个模型对应哪一批数据

## 4. 环境准备

### 4.1 Python 依赖

仓库核心依赖在 `requirements.txt` 里，重点是：

- `transformers>=4.49.0`
- `trl>=0.15.2`
- `peft>=0.14.0`
- `datasets>=2.14.6`
- `accelerate`
- `tensorboard`

安装基础依赖：

```bash
pip install -r requirements.txt
```

如果你要跑 4bit / 8bit：

```bash
pip install bitsandbytes
```

如果你要开 FlashAttention，按你的 CUDA / PyTorch 环境单独安装。

### 4.2 GPU 建议

仓库本身支持：

- 单卡
- 多卡 `torchrun`
- DeepSpeed
- LoRA / QLoRA / 全参

实操建议：

- 先默认 LoRA 或 QLoRA，不要一开始上全参
- 先用 0.5B / 1.5B / 7B 跑通流程，再放大
- 显存不够优先降低 `per_device_train_batch_size`
- 再调大 `gradient_accumulation_steps`
- 再考虑 `--load_in_4bit True --qlora True`

### 4.3 模型选择建议

在这个仓库里，如果你要做 SFT、DPO、RM、PPO 的完整流程，优先选标准 `AutoModelForCausalLM` / `AutoModelForSequenceClassification` 兼容的模型。

比较稳妥的是：

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- 更大的 Qwen instruct 系列

不建议你一开始用兼容性差的模型跑全链路。仓库 FAQ 已经明确提到，像 ChatGLM、Baichuan 在 RM / PPO 环节可能会受限。

## 5. 模板选择是第一处关键点

训练时要显式指定 `--template_name`，这决定了 prompt 如何被拼接。

例如仓库已经内置：

- `qwen`
- `vicuna`
- `alpaca`
- `chatglm3`
- `llama3`

如果你用 Qwen2.5 Instruct，优先用：

```bash
--template_name qwen
```

不要随便换模板。训练模板和数据风格不一致，最容易导致模型输出风格异常、loss 不稳定、推理效果发散。

## 6. 数据格式总表

### 6.1 SFT 数据格式

SFT 训练吃的是 ShareGPT 风格 jsonl，一行一个样本：

```json
{"conversations":[
  {"from":"human","value":"问题1"},
  {"from":"gpt","value":"回答1"}
]}
```

多轮也支持：

```json
{"conversations":[
  {"from":"human","value":"问题1"},
  {"from":"gpt","value":"回答1"},
  {"from":"human","value":"问题2"},
  {"from":"gpt","value":"回答2"}
]}
```

也支持第一条是 `system`：

```json
{"conversations":[
  {"from":"system","value":"你是一位严谨的医学助手。"},
  {"from":"human","value":"问题"},
  {"from":"gpt","value":"回答"}
]}
```

注意：

- 角色名必须是 `system` / `human` / `gpt`
- 第一轮如果不是 `human`，预处理时可能被跳过
- 对话必须成对，最后不能落单

训练前先校验：

```bash
python validate_jsonl.py --file_path ./data/custom/sft/train.jsonl
python validate_jsonl.py --file_path ./data/custom/sft/val.jsonl
```

### 6.2 偏好数据格式

RM / DPO / ORPO 用的是偏好对比数据，一行一个样本：

```json
{
  "system": "",
  "history": [],
  "question": "用户问题",
  "response_chosen": "更好的回答",
  "response_rejected": "更差的回答"
}
```

支持多轮历史：

```json
{
  "system": "你是一位严谨的医学助手。",
  "history": [["第一问","第一答"],["第二问","第二答"]],
  "question": "当前问题",
  "response_chosen": "优选答案",
  "response_rejected": "劣选答案"
}
```

注意：

- `history` 必须是二元对列表
- `chosen` / `rejected` 要针对同一个 `question`
- 最好保证差异是“质量差异”，不是“任务不同”

### 6.3 PPO 数据格式

PPO 复用了 SFT 数据风格。`ppo_training.py` 会从 `conversations` 中抽 prompt。

因此你可以直接用 SFT 数据，或者准备一个更偏“待优化提示集合”的数据集：

```json
{"conversations":[
  {"from":"human","value":"请回答这个医学问题"},
  {"from":"gpt","value":"参考答案，可有可无"}
]}
```

实操上，PPO 阶段更重要的是 prompt 覆盖面，而不是 gold answer 本身。

### 6.4 GRPO 数据格式

当前仓库里的 `grpo_training.py` 不是通用聊天 RL，而是更偏规则奖励任务。它要求数据至少有：

```json
{"question":"题目", "answer":"标准答案"}
```

而且默认 reward 函数会检查输出是否包含：

- `<think>...</think>`
- `<answer>...</answer>`

并对答案正确性打分。

这意味着：

- 你如果做标准医疗对话，当前 `GRPO` 不能直接拿来用
- 想用于医学问答，要先改 `grpo_training.py` 的 reward 逻辑
- `run_grpo.sh` 里默认 `--train_file_dir data/grop` 还是个路径拼写问题，使用前要改成正确目录

所以如果你的目标是医疗问答对齐，优先顺序应该是：

1. SFT
2. DPO / ORPO
3. RM + PPO
4. 最后再考虑自己改 GRPO

## 7. 自己造数据怎么做

### 7.1 先定义你要优化的能力

不要一上来就“收集医学数据”。先拆任务。

建议按能力拆 4 类：

1. 医学问答准确性
2. 问诊追问能力
3. 风险控制和安全拒答
4. 表达风格与结构化输出

然后每类能力分别构造：

- SFT 正样本
- 偏好对比样本
- 评测集

### 7.2 SFT 数据怎么造

SFT 数据只负责“教会模型怎么答”，不负责精细偏好排序。

一条好 SFT 样本至少要满足：

- 问题边界清楚
- 回答结构稳定
- 不引入明显医学幻觉
- 需要时明确提示“建议就医”“不能替代医生面诊”

建议把 SFT 样本拆成几种来源：

- 人工编写高质量模板样本
- 业务历史问答清洗后的样本
- 让更强的 teacher model 生成，再人工审核
- 公开数据转换后的样本

仓库里已有角色对话生成脚本在 `role_play_data/`，可以作为“批量生成医患多轮对话”的参考思路，但你最终还是要人工筛。

### 7.3 偏好数据怎么造

偏好数据不是再写一遍 SFT，而是让模型学会区分“哪个回答更好”。

推荐构造来源：

1. 用 base model 对同一 prompt 采样多版回答
2. 用 SFT model 对同一 prompt 采样多版回答
3. 人工写一个强答案，再保留一个弱答案
4. 对 teacher / student / base 输出做两两比较

一个可操作的方法是：

1. 准备一批 prompt
2. 分别让 `base`、`SFT`、`不同温度采样的 SFT` 生成回答
3. 人工挑 `chosen` 和 `rejected`
4. 整理成偏好 jsonl

偏好标注优先规则建议固定下来，比如：

- 准确性优先于文采
- 安全性优先于完整性
- 回答切题优先于篇幅
- 能明确表达不确定性优先于乱猜

### 7.4 如何避免“自己造数据把模型带歪”

这是最容易踩坑的地方。

建议强制执行这些约束：

- 不把明显错误答案放进 SFT 正样本
- 不把风格差异误标成质量差异
- 不把超出模型能力边界的问题全部要求“确定回答”
- 不让偏好数据只覆盖一种题型
- 不让训练集和评测集 prompt 重合

### 7.5 数据转换脚本什么时候用

如果你拿到的是 Alpaca / QA 风格数据，可以先转换：

```bash
python convert_dataset.py \
  --in_file ./raw/medical_qa.json \
  --out_file ./data/custom/sft/train.jsonl \
  --data_type qa \
  --file_type json
```

常见场景：

- 已有 `input/output` 二列，用 `qa`
- 已有 `instruction/input/output` 三列，用 `alpaca`
- 已经是 `conversations`，就不用转

## 8. 先跑最小可用实验

不要直接全量训练。先跑通最小闭环。

建议第一轮只用：

- SFT: 500 到 2000 条
- 偏好数据: 200 到 1000 条
- eval: 100 到 300 条独立问题

目标不是出最优结果，而是确认：

- 数据格式没问题
- 模板没问题
- 训练能正常收敛
- 推理输出风格符合预期

## 9. SFT 实操

### 9.1 什么时候跑 SFT

只要你有“问答示范数据”，就先跑 SFT。它是后续一切对齐方法的起点。

### 9.2 LoRA SFT 推荐起步命令

下面给你一份更适合自己实验的模板，不直接用仓库默认样例路径：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file_dir ./data/custom/sft \
  --validation_file_dir ./data/custom/sft \
  --template_name qwen \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples -1 \
  --max_eval_samples 200 \
  --model_max_length 4096 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_strategy steps \
  --save_steps 200 \
  --eval_strategy steps \
  --eval_steps 100 \
  --logging_steps 10 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --output_dir ./outputs/sft/qwen2.5-1.5b-med-sft-v1 \
  --cache_dir ./cache
```

如果显存很紧，可以改成 QLoRA：

```bash
--qlora True --load_in_4bit True
```

### 9.3 SFT 最该调哪些参数

优先关注：

- `--template_name`
- `--model_max_length`
- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--num_train_epochs`
- `--use_peft`
- `--qlora`

参数建议：

- 数据质量高、量不大时，`epoch` 可以相对高一点
- 数据量大时，优先控制总步数，不要盲目多轮
- 先固定 `lora_rank=8/16`，不要一开始大量扫参
- 医疗问答如果需要更长上下文，再加 `model_max_length`

### 9.4 SFT 输出怎么看

输出目录里最重要的是：

- `adapter_model.*`
- `adapter_config.json`
- `checkpoint-*`
- `trainer_state.json`
- `logs/`

先看：

- 训练 loss 是否稳定下降
- eval loss 是否同步改善
- 是否很快过拟合

然后立刻抽样推理，不要只看 loss。

### 9.5 SFT 后一定要做的事：合并 LoRA

如果后面你要接 DPO、RM、PPO，建议把当前阶段 adapter 先合并掉。

```bash
python merge_peft_adapter.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct \
  --lora_model ./outputs/sft/qwen2.5-1.5b-med-sft-v1 \
  --output_dir ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged
```

这是因为这个仓库下一个阶段通常希望 `model_name_or_path` 指向一个可以直接 `from_pretrained` 加载的完整模型目录。

## 10. DPO / ORPO 实操

### 10.1 什么时候先做 DPO

如果你已经有：

- 一个能用的 SFT 模型
- 一批质量不错的偏好数据

那优先做 DPO。原因很简单：

- 路径短
- 比 PPO 稳
- 数据闭环更好做
- 代码和调参成本更低

### 10.2 DPO 起步命令

```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --template_name qwen \
  --train_file_dir ./data/custom/reward \
  --validation_file_dir ./data/custom/reward \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples -1 \
  --max_eval_samples 200 \
  --max_steps 1000 \
  --eval_steps 100 \
  --save_steps 200 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --device_map auto \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --output_dir ./outputs/dpo/qwen2.5-1.5b-med-dpo-v1 \
  --cache_dir ./cache
```

注意：

- `model_name_or_path` 最好指向合并后的 SFT 模型
- DPO 数据目录不要混入 SFT 的 `conversations` 文件

### 10.3 ORPO 什么时候用

ORPO 可以当成 DPO 的一个平行实验分支。你不需要一开始就用它替代 DPO，更合理的是把它当对照组。

建议实验设计：

1. 同一份 SFT merged model
2. 同一份 preference 数据
3. 分别跑 DPO 和 ORPO
4. 比较输出质量、训练稳定性、显存、速度

### 10.4 DPO / ORPO 之后也要合并 LoRA

```bash
python merge_peft_adapter.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --tokenizer_path ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --lora_model ./outputs/dpo/qwen2.5-1.5b-med-dpo-v1 \
  --output_dir ./outputs/merged/qwen2.5-1.5b-med-dpo-v1-merged
```

## 11. RM + PPO 实操

### 11.1 什么时候值得上 PPO

只有在下面条件同时满足时，PPO 才值得花时间：

- 你已经有比较稳定的 SFT 基线
- 你已经有足够好的偏好数据
- 你愿意调更多训练细节
- 你接受训练成本和不稳定性明显高于 DPO

如果还没有，优先把 SFT 和 DPO 做扎实。

### 11.2 先训练 RM

RM 用的是同一份偏好数据：

```bash
CUDA_VISIBLE_DEVICES=0,1 python reward_modeling.py \
  --model_name_or_path ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --train_file_dir ./data/custom/reward \
  --validation_file_dir ./data/custom/reward \
  --template_name qwen \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples -1 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 0.001 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bf16 \
  --torch_dtype bfloat16 \
  --device_map auto \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --output_dir ./outputs/rm/qwen2.5-1.5b-med-rm-v1
```

### 11.3 合并 RM LoRA

PPO 脚本里 RM 是通过 `AutoModelForSequenceClassification.from_pretrained` 直接加载的，所以建议先把 RM adapter 合并成完整模型。

```bash
python merge_peft_adapter.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --tokenizer_path ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --lora_model ./outputs/rm/qwen2.5-1.5b-med-rm-v1 \
  --output_dir ./outputs/merged/qwen2.5-1.5b-med-rm-v1-merged
```

### 11.4 再跑 PPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python ppo_training.py \
  --sft_model_path ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --reward_model_path ./outputs/merged/qwen2.5-1.5b-med-rm-v1-merged \
  --template_name qwen \
  --train_file_dir ./data/custom/ppo \
  --validation_file_dir ./data/custom/ppo \
  --max_source_length 1024 \
  --response_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --do_train \
  --total_episodes 30000 \
  --num_train_epochs 1 \
  --missing_eos_penalty 1.0 \
  --eval_strategy steps \
  --eval_steps 100 \
  --report_to tensorboard \
  --output_dir ./outputs/ppo/qwen2.5-1.5b-med-ppo-v1
```

### 11.5 PPO 的实操提醒

- PPO 不适合在脏数据上硬跑
- PPO 的奖励模型质量直接决定优化方向
- 先小步试验，不要一开始就长时间训练
- PPO 最好和 DPO 做平行对比，而不是盲信 PPO 一定更强

## 12. PT 是否要做

项目支持增量预训练 `pretraining.py` / `run_pt.sh`，但这里要讲清楚：

- PT 是可选项
- PT 很吃语料质量
- PT 很吃算力
- 如果你的领域知识已被 base model 学过很多，收益可能并不明显

适合做 PT 的情况：

- 你有大量高质量、未被模型充分覆盖的领域原始文本
- 你做的是明显专域知识迁移
- 你准备承受更长训练时间

不适合时，就直接从 SFT 开始。

## 13. 推理验证怎么做

### 13.1 交互式人工抽查

```bash
python inference.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-dpo-v1-merged \
  --interactive
```

如果你还没合并，而只是想临时看 LoRA：

```bash
python inference.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_model ./outputs/sft/qwen2.5-1.5b-med-sft-v1 \
  --interactive
```

### 13.2 批量推理

准备一个文本文件，每行一个 prompt：

```text
孩子发烧怎么办？
白带增多发黄是什么原因？
请给出问诊时需要进一步确认的关键问题。
```

然后跑：

```bash
python inference.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-dpo-v1-merged \
  --data_file ./data/custom/eval/prompts.txt \
  --output_file ./outputs/predictions/dpo_eval.jsonl
```

这一步是做 base 对比实验的基础。

## 14. 自己和 base 跑对比实验，应该怎么做

这是你真正要长期坚持的部分。

### 14.1 至少保留 4 个模型快照

建议最少保留：

1. `base`
2. `sft_merged`
3. `dpo_merged` 或 `orpo_merged`
4. `ppo_model` 或最终版本

不要只保留“最后那个最好模型”，否则你根本没法回答到底是哪一步带来了收益。

### 14.2 固定一份独立评测集

评测集不要参与训练，也不要用在偏好标注中。

建议单独做 3 套：

1. `general_med_qa`
2. `safety_and_refusal`
3. `dialog_followup`

每套 100 到 300 条，足够先形成稳定观察。

### 14.3 对比实验的最小流程

1. 用同一份 `prompts.txt`
2. 分别跑 `base` / `sft` / `dpo` / `ppo`
3. 把输出保存成不同文件
4. 人工评审或脚本打分
5. 记录结果到实验表

例如：

```bash
python inference.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --data_file ./data/custom/eval/prompts.txt \
  --output_file ./outputs/predictions/base_eval.jsonl
```

```bash
python inference.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --data_file ./data/custom/eval/prompts.txt \
  --output_file ./outputs/predictions/sft_eval.jsonl
```

```bash
python inference.py \
  --base_model ./outputs/merged/qwen2.5-1.5b-med-dpo-v1-merged \
  --data_file ./data/custom/eval/prompts.txt \
  --output_file ./outputs/predictions/dpo_eval.jsonl
```

### 14.4 评测维度怎么定

医疗场景建议至少看：

- 准确性
- 安全性
- 切题程度
- 是否有过度自信幻觉
- 是否会主动追问关键信息
- 是否给出合理就医建议
- 格式稳定性

如果你自己做人工对比，推荐采用：

- 5 分制逐项打分
- 或 A/B 偏好投票

### 14.5 建议你维护一张实验总表

字段至少包括：

- 实验编号
- base model
- template
- 训练数据版本
- 偏好数据版本
- 训练方法
- 关键超参
- 输出目录
- 主观评测结果
- 备注

你后面迭代会非常依赖这张表。

## 15. 推荐的实验推进顺序

### 阶段 A：先跑通

1. 小样本 SFT
2. 小样本 DPO
3. base / sft / dpo 三模型人工对比

### 阶段 B：把数据做好

1. 扩充 SFT 高质量样本
2. 系统化制作偏好数据
3. 固定评测集

### 阶段 C：做方法对比

1. SFT only
2. SFT + DPO
3. SFT + ORPO
4. SFT + RM + PPO

### 阶段 D：做资源优化

1. LoRA vs QLoRA
2. 0.5B / 1.5B / 7B 对比
3. 长上下文配置对比

## 16. 日志和可视化

所有训练阶段都建议开 TensorBoard：

```bash
tensorboard --logdir ./outputs --host 0.0.0.0 --port 8008
```

重点看：

- train loss
- eval loss
- learning rate
- 是否突然震荡

但注意，loss 不是最终效果。医疗场景下，人工抽查和独立评测更重要。

## 17. 常见坑

### 17.1 模板不一致

表现：

- 输出格式奇怪
- 模型学不会角色边界
- 推理风格异常

处理：

- base model 是什么，就选对应模板
- 训练和后续实验尽量保持一致

### 17.2 数据目录混用

表现：

- SFT 时把 reward 数据也读进来
- DPO 时误读 `conversations` 文件

处理：

- SFT / reward / ppo / eval 分目录

### 17.3 LoRA 不合并就直接进下一阶段

这个项目里，下一阶段很多地方默认用 `from_pretrained` 直接加载模型目录，因此 LoRA 阶段结束后建议先 merge，再往下走。

### 17.4 只看 loss，不看生成

医疗场景里最危险的是：

- loss 漂亮
- 输出却有错误医学建议

所以每次训练后必须做人工抽检。

### 17.5 PPO 过早上量

PPO 很容易因为 RM 质量、采样分布、超参设置而不稳定。没有稳定 SFT / DPO 基线前，不要急着重投入 PPO。

### 17.6 GRPO 误用

当前仓库里的 GRPO 逻辑更适合规则可验证任务，不是开箱即用的医疗对话 RL。

## 18. 一个你现在就可以执行的最小落地方案

如果你现在就要开工，建议这么干：

1. 选 `Qwen/Qwen2.5-1.5B-Instruct`
2. 准备 1000 到 5000 条高质量医学 SFT 数据
3. 跑 LoRA SFT
4. 合并模型
5. 准备 500 到 2000 条偏好数据
6. 跑 DPO
7. 合并模型
8. 固定一套 200 条独立评测集
9. 对比 `base vs sft vs dpo`
10. 只有当 DPO 收益明确后，再决定是否继续做 RM + PPO

这条路线最稳，也最符合这个仓库当前脚本的成熟度。

## 19. 你在这个项目里应该怎么理解“完整实验闭环”

完整闭环不是“把所有脚本都跑一遍”，而是下面这件事：

1. 明确任务目标
2. 做出对应的训练数据
3. 跑出一个可复现实验
4. 留下阶段性模型快照
5. 用独立评测集和 base 做对比
6. 根据失败案例再回改数据和训练策略

真正决定结果的，通常不是某个算法名，而是：

- 数据质量
- 任务拆分是否清楚
- 实验记录是否规范
- 对比实验是否严谨

## 20. 建议你接下来立刻做的事

1. 建好 `data/custom/{sft,reward,ppo,eval}` 目录
2. 先整理一版高质量 SFT 数据和独立评测集
3. 先跑一轮小样本 SFT
4. 合并后做 base 对比
5. 再开始做偏好数据和 DPO

如果你后面要，我可以继续直接帮你做两件事：

1. 按这个仓库的格式，给你设计一套“医学 SFT 数据 schema + 偏好数据标注规范”
2. 直接在仓库里补一套“base / sft / dpo 自动批量对比脚本”
