# MedicalGPT 服务器连接与环境配置完整指南

这份文档的目标很直接：

指导你从本地连接服务器，在服务器上找到自己的工作目录，部署项目，配置 Python 虚拟环境，安装依赖，并做训练前检查。

适合当前这个项目，也适合你后续在 GPU 服务器上长期迭代。

## 1. 总体流程

建议你固定按这个顺序做：

1. 连上服务器
2. 找到自己的个人工作目录
3. 确认 Python / CUDA / GPU 基础环境
4. 拉取或上传项目代码
5. 创建虚拟环境
6. 安装依赖
7. 检查关键库是否可用
8. 跑最小推理或训练测试
9. 再开始正式实验

不要反过来。很多问题都不是代码问题，而是目录、权限、Python 版本、CUDA 兼容性问题。

## 2. 连接服务器前你需要准备什么

至少确认这几件事：

- 服务器地址
- SSH 端口
- 用户名
- 登录方式：密码或私钥
- 服务器是否需要 VPN
- 服务器是否需要跳板机

你这类内网服务器最常见的问题不是密码错，而是：

- 没连 VPN
- 不在公司网络
- SSH 端口不是 22
- 服务器只能从堡垒机进入

## 3. 本地如何连接服务器

### 3.1 使用密码连接

```bash
ssh liangfuquan@10.176.22.62
```

如果端口不是默认 `22`：

```bash
ssh -p 2222 liangfuquan@10.176.22.62
```

### 3.2 使用私钥连接

先确保私钥权限正确：

```bash
chmod 400 /path/to/private_key
```

然后连接：

```bash
ssh -i /path/to/private_key liangfuquan@10.176.22.62
```

### 3.3 第一次连接时的提示

第一次连接通常会看到主机指纹确认：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

## 4. 登录后第一件事：找到你的个人工作目录

先不要急着装环境。先确认你当前用户的目录结构。

登录后执行：

```bash
echo "USER=$USER"
echo "HOME=$HOME"
pwd
ls -la ~
```

然后继续找你常用工作目录：

```bash
find ~ -maxdepth 2 -type d \( -name work -o -name workspace -o -name projects -o -name code -o -name MedicalGPT -o -name medical-gpt \) 2>/dev/null | sort
```

### 4.1 怎么判断哪个目录最适合放项目

优先选这种目录：

- 在你自己的 `HOME` 下
- 你有完整读写权限
- 容量足够
- 不会和别人共用

常见合适位置：

- `~/work`
- `~/workspace`
- `~/projects`
- `~/code`

如果都没有，可以自己建：

```bash
mkdir -p ~/workspace
```

## 5. 检查服务器基础环境

先看操作系统和 Python：

```bash
uname -a
cat /etc/os-release 2>/dev/null || sw_vers 2>/dev/null
python3 --version
which python3
pip3 --version
```

如果是 GPU 服务器，再看显卡和 CUDA：

```bash
nvidia-smi
nvcc --version 2>/dev/null
```

你要关注：

- 有没有 GPU
- 驱动是否正常
- Python 版本是不是太老
- CUDA 版本和你准备安装的 PyTorch 是否匹配

### 5.1 对这个项目的 Python 建议

建议优先：

- Python `3.10`
- 或 Python `3.11`

如果服务器只有 `3.8`，通常也能凑合，但后续库兼容性会差一些。

## 6. 把项目放到服务器上

你有两种方式。

### 6.1 方式一：服务器直接 `git clone`

如果服务器能访问 GitHub：

```bash
cd ~/workspace
git clone https://github.com/LFrankl/medical-gpt.git
cd medical-gpt
```

如果你是推到私有仓库，就要保证服务器上的 Git 凭证可用。

### 6.2 方式二：本地上传到服务器

如果服务器不能直接访问 GitHub，可以用 `scp` 或 `rsync`。

#### `scp` 方式

```bash
scp -r /path/to/MedicalGPT liangfuquan@10.176.22.62:~/workspace/medical-gpt
```

#### `rsync` 方式

```bash
rsync -avz /path/to/MedicalGPT/ liangfuquan@10.176.22.62:~/workspace/medical-gpt/
```

如果需要指定 SSH key：

```bash
rsync -avz -e "ssh -i /path/to/private_key" /path/to/MedicalGPT/ liangfuquan@10.176.22.62:~/workspace/medical-gpt/
```

### 6.3 上传后先确认目录

```bash
cd ~/workspace/medical-gpt
pwd
ls
```

## 7. 创建虚拟环境

### 7.1 先确认 `venv` 可用

```bash
python3 -m venv --help >/dev/null && echo "venv ok"
```

### 7.2 创建虚拟环境

在项目根目录下执行：

```bash
cd ~/workspace/medical-gpt
python3 -m venv .venv
```

### 7.3 激活虚拟环境

```bash
source .venv/bin/activate
```

激活后你应该能看到终端前面多了类似：

```text
(.venv)
```

然后确认：

```bash
which python
python --version
```

### 7.4 以后每次进项目都这样做

```bash
cd ~/workspace/medical-gpt
source .venv/bin/activate
```

## 8. 安装依赖

### 8.1 先升级基础工具

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 8.2 安装项目依赖

```bash
pip install -r requirements.txt
```

### 8.3 如果你要做量化训练

这个项目如果要用 4bit / 8bit，通常还需要：

```bash
pip install bitsandbytes
```

注意：

- `bitsandbytes` 对 CUDA、Linux 和显卡环境有要求
- Mac 本地一般别装这个做训练
- 服务器上如果装不上，先不要阻塞整个环境配置

### 8.4 如果你要用 FlashAttention

这个库安装和 CUDA / 编译环境强绑定，建议在确认 `torch` 和 CUDA 都没问题后再单独处理，不要一上来就混在一起装。

## 9. 安装 PyTorch 的建议

这个项目依赖 `transformers`、`trl`、`peft`，但最敏感的是 `torch`。

先检查服务器上是否已有可用版本：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
```

如果没装或不兼容，再按服务器 CUDA 版本安装官方对应轮子。

例如常见 Linux CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你不确定 CUDA 版本，先不要乱装。

## 10. 对这个项目做最小可用性检查

进入项目目录并激活虚拟环境后，先执行：

```bash
python - <<'PY'
import transformers
import datasets
import peft
import trl
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
PY
```

然后检查 GPU：

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

这两步通过，才说明基础环境大体可用。

## 11. 项目级检查

### 11.1 先看几个关键脚本是否存在

```bash
ls supervised_finetuning.py
ls dpo_training.py
ls reward_modeling.py
ls ppo_training.py
ls inference.py
```

### 11.2 检查新加的工具和文档

```bash
ls compare_batch_models.py
ls docs
ls data/custom_examples
```

## 12. 最小推理测试

如果你先只想确认模型和环境能不能跑，不要一上来训练。先推理。

比如：

```bash
python inference.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --interactive
```

如果服务器不能联网下载模型，你就需要先把模型权重也同步到服务器，或者使用本地已有模型目录。

## 13. 最小训练测试

先用仓库里的小样例数据做一个极小规模 SFT 冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file_dir ./data/finetune \
  --validation_file_dir ./data/finetune \
  --template_name qwen \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 20 \
  --max_eval_samples 5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --output_dir ./outputs/smoke-sft \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to none
```

目标不是出效果，而是验证：

- 模型能加载
- 数据能读
- 训练流程能启动

## 14. 如果你准备正式训练

在正式训练前，建议先做：

1. 确定项目目录
2. 确定模型目录
3. 确定数据目录
4. 确定输出目录
5. 确定日志目录

建议结构：

```text
~/workspace/medical-gpt/
~/workspace/models/
~/workspace/data/
~/workspace/outputs/
```

这样不要把：

- 代码
- 模型
- 训练输出
- 数据

全都堆在一个目录里。

## 15. 建议配置 `tmux`

训练时强烈建议用 `tmux`，不要直接在普通 SSH 会话里跑。

### 15.1 启动新会话

```bash
tmux new -s medgpt
```

### 15.2 断开但不结束

按：

```text
Ctrl-b d
```

### 15.3 重新进入

```bash
tmux attach -t medgpt
```

### 15.4 查看已有会话

```bash
tmux ls
```

如果服务器没装 `tmux`，尽量让管理员装，或者你自己用系统包管理器安装。

## 16. 建议配置环境变量

你如果后面要长期在服务器上训练，API key、HF cache 等最好统一配置。

例如在服务器的 `~/.zshrc` 或 `~/.bashrc` 里加：

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

如果你有 API key：

```bash
export AICODING_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
```

改完后执行：

```bash
source ~/.zshrc
```

或者：

```bash
source ~/.bashrc
```

## 17. 常见问题

### 17.1 `ssh: connect to host ... port 22: Operation timed out`

通常说明：

- 没连 VPN
- 不在内网
- SSH 端口不对
- 被防火墙拦了

### 17.2 `Permission denied (publickey,password)`

说明：

- 用户名错了
- 私钥不对
- 私钥权限不对
- 服务器禁用了你当前的认证方式

### 17.3 `python3: command not found`

说明服务器没装 Python 3，或者不在 PATH 里。先联系管理员或用系统包管理器安装。

### 17.4 `No module named venv`

说明 Python 的 `venv` 组件没装，需要补安装系统包。

### 17.5 `torch.cuda.is_available() == False`

说明：

- 你装的是 CPU 版 PyTorch
- CUDA 不匹配
- 驱动有问题
- 当前机器没有 GPU

### 17.6 `bitsandbytes` 安装失败

先不要阻塞环境搭建，先把基础版本跑通。量化训练放到第二阶段再解决。

## 18. 一套最推荐的服务器初始化命令

如果你已经登录到服务器，并确定目录是 `~/workspace/medical-gpt`，可以按这个顺序直接执行：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/LFrankl/medical-gpt.git
cd medical-gpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python - <<'PY'
import transformers, datasets, peft, trl
print("ok")
PY
```

## 19. 一套最推荐的排查命令

如果你碰到环境问题，先把下面这组命令结果保存下来：

```bash
whoami
hostname
pwd
echo $HOME
python3 --version
which python3
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

这组输出基本能判断 80% 的问题。

## 20. 最后建议

真正的服务器部署，不要一开始就追求：

- 全参训练
- 大模型
- 长上下文
- 量化 + FlashAttention + DeepSpeed 一起上

先把最小闭环跑通：

1. SSH 正常
2. 目录清楚
3. 虚拟环境 OK
4. 依赖 OK
5. GPU OK
6. 推理 OK
7. 小样本训练 OK

这条线跑通以后，你再去加复杂配置，出问题也更容易定位。
