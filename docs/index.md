# MedicalGPT 训练文档中心

欢迎来到 MedicalGPT 训练文档中心。本页面汇总了所有训练相关的文档，帮助你从零开始完成大模型训练实验。

---

## 🚀 快速开始

如果你是第一次使用本项目，建议按以下顺序阅读：

1. [服务器连接与环境配置指南](./server_setup_guide.md) - 配置服务器环境
2. [实践路线图（双 4090）](./practice_roadmap_dual4090.md) - 了解推荐学习路线
3. [完整实验操作手册](./step_by_step_experiment_guide.md) - 手把手完成第一次实验

---

## 📚 文档导航

### 环境与准备
- [服务器连接与环境配置完整指南](./server_setup_guide.md)
  - SSH 连接、Python 环境、GPU 检查、依赖安装
  - 适合第一次在服务器上部署项目

- [实践路线图（双 4090 / 100G+ 内存）](./practice_roadmap_dual4090.md)
  - 推荐学习顺序：SFT → DPO → 部署
  - 硬件配置建议、时间规划

### 完整实验流程
- [完整实验操作手册：从零到部署的每一步](./step_by_step_experiment_guide.md)
  - 数据准备、SFT 训练、DPO 训练、推理验证、API 部署
  - 包含完整命令、监控方法、常见问题解决

- [服务器全流程指南：SFT 前置 + DPO 训练](./dpo_full_pipeline_server.md)
  - 双 4090 服务器上的完整 SFT → DPO 流程
  - 包含后台运行、日志监控

- [DPO 训练指南（双 4090 服务器）](./dpo_dual_4090_server_guide.md)
  - DPO 训练的详细说明
  - 参数配置、数据格式

### 数据准备
- [数据集说明](./datasets.md)
  - 支持的数据格式（ShareGPT、Alpaca 等）
  - 数据集下载与转换

- [数据标注与格式规范](./data_schema_and_labeling_guide.md)
  - SFT、DPO、RM 数据格式详解
  - 数据质量标准

- [LLM 数据生成手册](./llm_data_generation_playbook.md)
  - 使用 LLM 生成训练数据
  - 数据增强技巧

### 训练方法详解
- [微调与强化学习操作指南](./finetune_rl_operation_guide.md)
  - SFT、DPO、PPO、GRPO 等方法对比
  - 各方法的适用场景

- [训练细节说明](./training_details.md)
  - 训练原理、损失函数
  - 技术细节

- [训练参数说明](./training_params.md)
  - 各参数含义与推荐值
  - 参数调优建议

### 脚本使用
- [脚本概览（双 4090）](./scripts_overview_dual4090.md)
  - 双 4090 服务器推荐的脚本配置
  - run_sft.sh、run_dpo.sh 等

- [脚本概览（通用）](./scripts_overview.md)
  - 所有训练脚本的说明
  - 参数配置示例

### 其他
- [词表扩展](./extend_vocab.md)
  - 如何扩展模型词表
  - 适用于特定领域词汇

- [常见问题 FAQ](./FAQ.md)
  - 常见错误与解决方案

---

## 🎯 推荐学习路径

### 路径 1：完全新手（第一次训练大模型）
```
服务器环境配置 → 实践路线图 → 完整实验操作手册 → 开始第一次 SFT
```

### 路径 2：有基础（做过 SFT，想学 DPO）
```
DPO 训练指南 → 服务器全流程指南 → 数据标注规范 → 开始 DPO 实验
```

### 路径 3：深入研究（想理解原理和调参）
```
训练细节说明 → 训练参数说明 → 微调与强化学习操作指南 → 实验对比
```

---

## 💡 核心概念速查

| 概念 | 说明 | 相关文档 |
|------|------|---------|
| **SFT** | 监督微调，让模型学习领域知识 | [完整实验操作手册](./step_by_step_experiment_guide.md) |
| **DPO** | 直接偏好优化，让模型学习人类偏好 | [DPO 训练指南](./dpo_dual_4090_server_guide.md) |
| **LoRA** | 低秩适配，高效微调方法 | [训练细节说明](./training_details.md) |
| **QLoRA** | 量化 LoRA，进一步降低显存需求 | [训练参数说明](./training_params.md) |
| **ShareGPT** | 多轮对话数据格式 | [数据集说明](./datasets.md) |
| **Merge** | 合并 LoRA 和基座模型 | [完整实验操作手册](./step_by_step_experiment_guide.md) |

---

## 🔧 快速命令参考

### 数据验证
```bash
python validate_jsonl.py --file_path data/finetune/sharegpt_zh_1K_format.jsonl
```

### SFT 训练（双 4090）
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --train_file_dir ./data/finetune \
  --output_dir outputs-sft
```

### DPO 训练（双 4090）
```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
  --model_name_or_path merged-sft-model \
  --train_file_dir ./data/reward \
  --output_dir outputs-dpo
```

### 合并模型
```bash
python merge_peft_adapter.py \
  --base_model /data/models/Qwen2.5-7B-Instruct \
  --lora_model outputs-sft \
  --output_dir merged-model
```

### 推理测试
```bash
python inference.py --base_model merged-model --interactive
```

### 启动 API
```bash
python openai_api.py --model_name_or_path merged-model --port 8000
```

---

## 📊 硬件配置建议

| 配置 | 推荐模型 | 训练方法 | 预期效果 |
|------|---------|---------|---------|
| 单 4090 (24GB) | Qwen2.5-7B | QLoRA | 可训练，batch size 较小 |
| 双 4090 (48GB) | Qwen2.5-7B | QLoRA | 推荐配置，训练流畅 |
| 双 4090 (48GB) | Qwen2.5-14B | QLoRA | 可训练，需调整参数 |
| 4×A100 (320GB) | Qwen2.5-72B | Full Fine-tune | 全参数训练 |

---

## 🤝 获取帮助

- **GitHub Issues**: [提交问题](https://github.com/LFrankl/medical-gpt/issues)
- **FAQ**: [常见问题解答](./FAQ.md)
- **文档反馈**: 欢迎提 PR 改进文档

---

**最后更新**: 2026-03-27
