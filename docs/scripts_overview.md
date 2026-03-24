# MedicalGPT 脚本总览（`*.py` / `*.sh`）

这份文档说明仓库内各脚本的用途，方便快速判断“该跑哪个脚本”。

## 1) 训练入口脚本（Shell）

| 脚本 | 作用 | 对应阶段 |
|---|---|---|
| `run_pt.sh` | 启动增量预训练（PT），默认 `torchrun` 多卡 + LoRA 参数示例 | Stage 1 PT |
| `run_sft.sh` | 启动监督微调（SFT），默认 `torchrun` 多卡 + LoRA | Stage 2 SFT |
| `run_sft_accelerate.sh` | 用 `accelerate launch` 启动 SFT（替代 `torchrun` 方案） | Stage 2 SFT |
| `run_full_sft.sh` | 启动全参 SFT（`--use_peft False`），并给出 deepspeed 示例 | Stage 2 SFT（全参） |
| `run_rm.sh` | 启动奖励模型训练（RM），输入偏好数据 | Stage 3 RM |
| `run_ppo.sh` | 启动 PPO 强化学习训练，使用 SFT+RM | Stage 4 RLHF(PPO) |
| `run_dpo.sh` | 启动 DPO 训练（直接偏好优化） | 偏好对齐 DPO |
| `run_orpo.sh` | 启动 ORPO 训练（偏好优化变体） | 偏好对齐 ORPO |
| `run_grpo.sh` | 启动 GRPO 训练（R1/推理型任务示例，含 QLoRA 配置） | 偏好/RL 变体 |
| `run_grpo_xm_test.sh` | GRPO 小规模测试脚本（快速验证配置） | 调试脚本 |
| `run_quant.sh` | 调用量化脚本，对模型做 4bit 量化并保存 | 量化 |
| `run_eval_quantize.sh` | 调用量化评估脚本，评估量化模型（PPL） | 量化评估 |
| `vllm_deployment.sh` | 启动 vLLM OpenAI 兼容服务并附带 curl 测试示例 | 部署 |

## 2) 训练核心脚本（Python）

| 脚本 | 作用 | 输入/输出（简） |
|---|---|---|
| `pretraining.py` | 增量预训练主程序（CLM），支持 LoRA/QLoRA/deepspeed | 输入预训练语料；输出 PT checkpoint |
| `supervised_finetuning.py` | SFT 主程序（Trainer 版），支持 LoRA/QLoRA/FlashAttn 等 | 输入 SFT 数据；输出 SFT checkpoint |
| `supervised_finetuning_accelerate.py` | SFT 主程序（Accelerate 版） | 同上 |
| `reward_modeling.py` | 奖励模型训练（SequenceClassification） | 输入偏好数据；输出 RM 模型 |
| `ppo_training.py` | PPO 训练脚本（TRL PPOTrainer） | 输入 SFT/RM + RL 数据；输出 PPO 模型 |
| `dpo_training.py` | DPO 训练脚本（TRL DPOTrainer） | 输入偏好数据；输出 DPO adapter/模型 |
| `orpo_training.py` | ORPO 训练脚本（TRL ORPOTrainer） | 输入偏好数据；输出 ORPO adapter/模型 |
| `grpo_training.py` | GRPO 训练脚本（TRL GRPOTrainer，偏数学/规则奖励场景） | 输入数据集；输出 GRPO 模型 |
| `template.py` | 对话模板注册与格式化核心组件（qwen/vicuna 等） | 被训练与推理脚本共同依赖 |

## 3) 推理与服务脚本

| 脚本 | 作用 | 场景 |
|---|---|---|
| `inference.py` | 单机推理脚本，支持交互和批量推理，可加载 LoRA | 本地/服务器快速验证 |
| `inference_multigpu_demo.py` | 多卡推理示例（`torchrun`） | 多 GPU 推理演示 |
| `gradio_demo.py` | Gradio Web Demo | 可视化聊天页面 |
| `fastapi_server_demo.py` | 简化版 FastAPI 服务接口（`/chat`） | 轻量 API 服务 |
| `openai_api.py` | OpenAI 协议兼容 API 服务（含流式/工具调用框架） | 对接 OpenAI SDK/生态 |
| `chatpdf.py` | 本地 RAG 问答 Demo（PDF 分块+检索+生成） | 文档问答演示 |

## 4) 模型处理与量化脚本

| 脚本 | 作用 | 典型用途 |
|---|---|---|
| `merge_peft_adapter.py` | 将 LoRA adapter 合并回 base model | 阶段切换、部署前固化模型 |
| `model_quant.py` | 使用 bitsandbytes 进行 4bit 量化并比较推理性能 | 生成量化模型 |
| `eval_quantize.py` | 评估量化模型在 jsonl 数据上的困惑度 | 对比量化前后效果 |
| `merge_tokenizers.py` | 合并词表（base tokenizer + 领域 sentencepiece + 可选 jieba/baichuan 词） | 扩词表方案 |
| `build_domain_tokenizer.py` | 从领域文本训练 sentencepiece tokenizer | 领域 tokenizer 构建 |

## 5) 数据处理与校验脚本

| 脚本 | 作用 | 说明 |
|---|---|---|
| `convert_dataset.py` | 将 Alpaca/QA/其他格式转换成 ShareGPT 格式 jsonl | SFT 前数据统一 |
| `validate_jsonl.py` | 校验 ShareGPT jsonl 基本结构合法性 | 快速查坏数据 |
| `docs/numina_cot_sharegpt.py` | 将 NuminaMath-CoT 数据集转 ShareGPT 格式 | 数学类 SFT 数据准备 |

## 6) 角色扮演数据生成脚本（`role_play_data/`）

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `role_play_data/role_generate.py` | 基于种子角色提示扩写角色设定（护士/患者） | OpenAI API |
| `role_play_data/roleplay_data_generate_gpt4.py` | 用 GPT 生成多轮护士-患者对话数据 | OpenAI API |
| `role_play_data/roleplay_data_generate_doubao.py` | 用豆包（火山引擎）接口生成多轮角色对话 | 火山方舟 API |

## 7) 推荐使用顺序（最常见）

1. `pretraining.py`（可选，做领域增量 PT）
2. `supervised_finetuning.py`（SFT）
3. `dpo_training.py` 或 `orpo_training.py`（偏好对齐）
4. `merge_peft_adapter.py`（合并 LoRA）
5. `inference.py` / `openai_api.py` / `vllm_deployment.sh`（推理与部署）

## 8) 常见对应关系速查

- 想跑 DPO：`run_dpo.sh` -> `dpo_training.py`
- 想跑 SFT：`run_sft.sh` 或 `run_sft_accelerate.sh` -> `supervised_finetuning*.py`
- 想从 LoRA 变成可直接部署模型：`merge_peft_adapter.py`
- 想量化：`run_quant.sh` -> `model_quant.py`，评估用 `run_eval_quantize.sh` -> `eval_quantize.py`
- 想开 API 服务：`openai_api.py`（OpenAI 协议）或 `fastapi_server_demo.py`（简化接口）

