# openpi 中文导读：Physical Intelligence 的 π0 官方开源实现

> 配套《[具身智能全景指南](../具身智能全景指南.md)》第 8 章（VLA）与 8.1 节第五级。
> 结构基于 2026-10 主分支快照。

## 1. openpi 是什么

Physical Intelligence（π 系列论文的出品方）的官方开源库，包含 **π0**（arXiv:2410.24164）与 **π0-FAST**（arXiv:2501.09747）的训练与推理实现。**读它 = 读 π0 论文的工程实现**，是理解"预训练 VLM 主干 + flow matching 动作专家"这一主流 VLA 架构最直接的材料。

与 LeRobot 的分工：LeRobot 是"库"（数据格式 + 硬件 + 多策略统一接口，内含 π0 的 PyTorch 移植权重）；openpi 是"官方训练源码"（JAX 为主，含完整预训练/微调管线）。学架构读 openpi，跑真机用 LeRobot。

## 2. 仓库结构地图

```text
openpi/
├── src/openpi/
│   ├── models/            ★ 核心：模型定义（JAX）
│   │   ├── pi0.py               π0 主模型：PaliGemma 主干 + flow matching 动作专家
│   │   ├── pi0_fast.py          π0-FAST：自回归离散动作 token（FAST tokenizer）
│   │   ├── gemma.py             Gemma/PaliGemma 语言-视觉主干封装
│   │   ├── siglip.py  vit.py    SigLIP 视觉塔
│   │   ├── lora.py              LoRA 微调（消费级显卡微调 π0 的关键）
│   │   └── tokenizer.py         状态/动作/图像 token 化
│   ├── models_pytorch/    PyTorch 移植（推理为主，与 LeRobot 生态互通）
│   ├── transforms.py      ★ 各本体的观测/动作变换（归一化、语义化 state）
│   ├── policies/          本体接入层：aloha_policy / droid_policy / libero_policy
│   ├── training/          训练循环（JAX/TPU 优化、混合精度、checkpoint）
│   ├── serving/           推理服务（websocket server，边-云分离部署）
│   └── shared/            Array/normalize 等共享工具
├── examples/              aloha_real / aloha_sim / droid / libero / ur5 / simple_client
├── scripts/               数据转换（LeRobotDataset→openpi）、权重下载、微调入口
└── docs/                  微调教程与 URDF/相机标定说明
```

## 3. 三个必读文件（对应 π0 论文公式）

1. **`models/pi0.py`** —— 论文的核心：动作专家只在**单独的 flow matching 步骤**中与 VLM 主干交互；训练目标是从噪声到动作块的**向量场回归**（论文式 (4)），推理迭代 10 步积分。`sample_actions()` 即 Algorithm 2。
2. **`transforms.py`** —— 常被忽视但极其重要：每个本体把原始观测映射为统一语义空间（关节角→规范化 state、图像 resize、动作归一化）。**跨本体迁移的一半功劳在这里**（对应指南 8.2 "本体接口"）。
3. **`policies/policy_config.py`** —— 看懂"预训练 checkpoint + 本体 transforms + 推理配置"如何组装成一个可部署策略；微调自己的机器人从这里改起。

## 4. 微调路径（现实可操作的部分）

预训练（~1 万小时真机数据、多机多卡）对个人不可行，但 openpi 的设计让**LoRA 微调**成为个人/实验室的现实路径：

```text
选基座        pi0_libero（仿真操作）/ pi0_aloha / pi0_droid
准备数据      LeRobotDataset 格式（scripts/convert 数据转换脚本）
LoRA 微调     冻结主干，只训动作专家/低秩适配层（单卡 24G 可跑低配版）
部署          serving/ 起本地服务，examples/simple_client 连真机
```

算力参考：Libero 套件 LoRA 微调单张 RTX 4090 级即可；全参数微调需 A100×多卡。

## 5. 与指南概念的对查表

| openpi 概念 | 指南对应 |
|---|---|
| flow matching 动作专家 | 8.1/8.2（动作头三分：自回归/扩散/流匹配） |
| π0-FAST 的 BPE 动作 token | 8.2（离散 vs 连续之争）、11.6 争论① |
| transforms 每本体变换 | 8.2（本体接口）、8.3（跨本体数据） |
| libero/aloha/droid policies | 4.4（仿真基准）、8.3（数据集） |
| serving 分离式推理 | 8.2（推理频率-质量权衡） |

---
*本文为社区中文导读，非官方文档；以 [openpi 仓库](https://github.com/Physical-Intelligence/openpi) 与 π0 论文为准，欢迎 PR 修正。*
