# sim2real-deploy

强化学习策略从仿真到实机的部署链路。计划每台机器人一个目录，训练侧与部署侧在结构上分离开。

## 设计原则

**训练、部署、数据管理分三个仓库。**

```
robots/<机器人>/
├── training/    依赖 isaaclab，装在训练环境
└── deploy/      零 isaaclab 依赖，装在部署环境
```
## 备忘
两者是**独立的 Python 包，各有 pyproject.toml**，在不同环境env_all_pro和d1_deploy_copy。部署环境里没有 `isaaclab`,python版本也不一样，避免「部署代码误引用了仿真框架」。

##管理方式

- 两侧之间**只允许**通过三样东西通信：策略权重文件、观测契约表、关节映射表
- 契约表由脚本从配置**自动导出**，不手动誊写
- 契约文件**进版本库**（尽管是生成物），因为行为异常时第一件事就是 `git diff` 
- sim2sim 与实机**共用同一份**观测拼装、关节映射、安全逻辑

## 结构

```
docs/contracts/          接缝契约（自动生成，进 git）
tools/                   跨机器人通用工具
robots/<机器人>/
├── assets.toml          资产路径、出处、转换命令、拍板参数
├── assets/              派生资产（USD、权重），gitignore
├── training/            Isaac Lab 任务定义
├── deploy/              部署运行时（可独立安装到机载计算机）
└── tools/               该机器人专用：导出、比对、验证
```

资产（URDF、网格、USD、权重）**不进版本库**，`assets.toml` 记录出处与再生成方法。

## 机器人

| 机器人 | 状态 |
|---|---|
| [`d1_edu`](robots/d1_edu/) — 智元 D1 edu 四足 | 进行中 | 已部署

## 环境

| 用途 | 环境 | 关键依赖 |
|---|---|---|
| 训练 | `env_all_pro` | Python 3.11、Isaac Sim 5.1、Isaac Lab 0.54.3、torch 2.7.0+cu128、rsl_rl |
| 部署验收 / sim2sim | `d1_deploy` | Python 3.11、torch 2.7.0 CPU、numpy、mujoco。**不装 isaaclab、不装 rsl_rl** |
| 机载 | 待定 | 见各机器人 `assets.toml` |

`d1_deploy` 环境是交付验收环境：策略应能在这个没有仿真框架的环境里加载成功。

##注意：附近有AI出没。
本项目有AI辅助，警惕AI表达风格。
