# LeRobot 中文导读：一张地图逛完 huggingface/lerobot

> 配套《[具身智能全景指南](../具身智能全景指南.md)》第 10.1 节（开源生态）与第 12 章（学习路线图）。
> 结构基于 2026-10 主分支快照；LeRobot 迭代很快，以仓库为准。

## 1. LeRobot 是什么

Hugging Face 出品的真机机器人学习全家桶：**统一数据格式 + 低成本硬件支持 + SoTA 策略实现 + 训练/评测工具链**。定位类似"NLP 的 transformers 库"之于机器人。ICLR 2026 论文：[arXiv:2602.22818](https://arxiv.org/abs/2602.22818)。

对中文学习者的意义：**它是当前入门真机具身智能成本最低的路径**——一台 ~¥1500 的 SO-101 双臂（3D 打印 + 舵机）+ 一张消费级 GPU 就能跑通"遥操作采数据 → 训策略 → 真机部署"全流程。

## 2. 仓库结构地图（src/lerobot/）

```text
lerobot/
├── robots/           机器人本体接口（统一 Robot 类：get_observation / send_action）
│   ├── so_follower/        SO-100/SO-101 双臂（入门首选，详见第 12.1 月计划）
│   ├── lekiwi/             轮式双臂（Mobile ALOHA 平替）
│   ├── koch_follower/      Koch v1.1 双臂
│   ├── unitree_g1/         宇树人形（指南第 9 章）
│   └── reachy2/ hope_jr/ omx/ earthrover/ ...
├── teleoperators/    遥操作主端（so_leader / koch_leader / gamepad / keyboard / phone / homunculus VR 手套）
├── cameras/          相机（OpenCV / RealSense / ...）
├── datasets/         LeRobotDataset v3：Parquet(状态/动作) + MP4(视频)，HF Hub 托管与流式加载
├── policies/         策略实现（见下方对照表）
├── rl/               强化学习（HIL-SERL 真机 RL）
├── rewards/          奖励模型（SARM 等）
├── envs/             仿真环境接口（LIBERO / MetaWorld 评测）
├── inference/        部署推理；remote_inference/ 为边-云分离推理
├── processor/        观测/动作预处理管线（transforms 组合式）
├── optim/ transport/ distributed/ streaming/   训练基建
└── scripts/          CLI 入口（lerobot-train / lerobot-record / lerobot-eval / lerobot-info）
```

**架构要点**（对应指南 2.5 节动作空间、8.2 节 VLA 解剖）：

- `Robot` 与 `Policy` 通过**特征契约**（observation/action 的 key、shape、fps）解耦——跨本体适配的工程答案；
- `processor/` 把"归一化、裁剪、mask"做成可组合 transform，与策略配置一起版本化——复现性的关键；
- 策略均实现 `select_action(obs)`，训练走 `lerobot-train --policy.type=X --dataset.repo_id=Y`。

## 3. policies/ 目录 ↔ 论文对照

| 目录 | 论文 | 指南章节 |
|---|---|---|
| `act/` | ACT / ALOHA, arXiv:2304.13705 | 7.3 第二代 |
| `diffusion/` | Diffusion Policy, arXiv:2303.04137 | 7.3 第二代 |
| `vqbet/` | VQ-BeT（离散 token 化策略） | 7.3 |
| `tdmpc/` | TD-MPC2, arXiv:2310.16828 | 附录 B 世界模型 |
| `smolvla/` | SmolVLA, arXiv:2506.01844（450M，消费级 GPU 可训） | 8.2 推理效率 |
| `pi0/` `pi05/` `pi0_fast/` | π0, arXiv:2410.24164；π0.5, arXiv:2504.16054；FAST, arXiv:2501.09747 | 8.1 第五级 |
| `groot/` | GR00T N1, arXiv:2503.14734 | 8.1 第五级 |
| `wall_x/` `wall_oss_05/` | Wall-OSS（银河通用）, arXiv:2509.11766 | 8.1 |
| `molmoact2/` | MolmoAct（AI2）, arXiv:2508.07917 | 8.5 时间线 |
| `rtc/` | Real-Time Chunking, arXiv:2506.07339 | 8.2 推理频率 |
| `vla_jepa/` | VLA-JEPA（JEPA 路线进机器人） | 8.4 一体化派 |
| 其余（eo1/ evo1/ xvla/ flux3/ dm05/ lingbot_*/ fastwam/ lawam/ ...） | 见 `docs/source/` 对应 mdx | — |

**读代码建议顺序**（对应指南 7.3 三代演进）：`act/policy_act.py`（最短，看懂 CVAE + action chunk）→ `diffusion/policy_diffusion.py`（生成式策略）→ `pi0/`（预训练 VLM + flow matching 动作头）→ `smolvla/`（自己训得动的 VLA）。

## 4. 半天上手路线

```bash
pip install lerobot
lerobot-info                      # 自检环境

# 1) 不买硬件先跑数据集
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
ds = D('lerobot/aloha_mobile_cabinet'); print(ds[0]['action'].shape)"

# 2) 云端/本地训练一个策略（仿真或公开数据集）
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# 3) 有 SO-101 后：遥操作采集 → 训练 → 真机回放
lerobot-record --robot.type=so_follower --teleop.type=so_leader \
               --dataset.repo_id=${HF_USER}/my_first_dataset
```

硬件预算参考：SO-101 双臂约 $100–250/台（主从两台即一套采集系统）；不需要机械臂也能用 `gamepad`/`phone` 遥操作部分设备。中文装配/采集/训练图文教程见 README「Resources」中同济子豪兄的飞书教程。

## 5. 常见坑

1. **视频解码依赖**：数据集加载依赖 ffmpeg，Windows 下建议 WSL2 或 conda 环境；
2. **主从臂标定**：`lerobot-calibrate` 每次重装机械臂后必须重跑，否则采集数据关节零点漂移；
3. **fps 一致性**：数据采集 fps、策略训练 fps、真机推理 fps 三者要一致（processor 会校验）；
4. **数据量预期**：桌面单任务 ACT 约 50 条演示起步（指南 7.3），别指望 5 条训出鲁棒策略。

---
*本文为社区中文导读，非官方文档；错误请以 [官方文档](https://huggingface.co/docs/lerobot/index) 为准，欢迎 PR 修正。*
