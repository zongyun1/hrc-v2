# AICR 远程连接与作业规则

本文件是本仓库使用 AICR 的操作清单，依据 [AICR 官方文档](https://docs.aicr.ai/) 整理。目标是让连接、同步和作业提交可重复，并避免影响共享登录节点或误删远程文件。

## 连接

- 使用本机 SSH 别名 `aicr`；其目标应为 `login.aicr.ai`。首次配置按 [SSH 文档](https://docs.aicr.ai/connecting/ssh/) 下载证书，并确认私钥权限为 `600`、证书为 `644`。
- 证书接近过期时，通过 OnDemand 的 SSH Certificate 应用重新下载；不要把私钥、证书或 `.passphrase` 提交到仓库。
- SSH 配置建议设置 `ServerAliveInterval 60`。登录失败先检查证书和网络，不要反复重试造成无意义连接。

```sshconfig
Host aicr
    HostName login.aicr.ai
    User AICR_USERNAME
    IdentityFile ~/.ssh/id_ed25519_aicr
    CertificateFile ~/.ssh/id_ed25519_aicr-cert.pub
    ServerAliveInterval 60
```

## 登录节点纪律

- 登录节点只做编辑、轻量检查、文件传输，以及 `sbatch`/`squeue`/`sacct`/`scancel` 等调度操作。
- 不在登录节点运行 Python、训练、渲染、编译、批量扫描或大规模 `pip`/`uv` 安装；计算必须通过 `sbatch` 或 `srun` 申请资源。
- 不能直接 SSH 到计算节点；有活动作业时，先用 `squeue -u $USER` 找到节点，再通过该作业连接。

## 作业提交与监控

- 先看资源和自己的作业：

```bash
ssh aicr 'sinfo'
ssh aicr 'squeue -u $USER'
ssh aicr 'sacct -u $USER --starttime today'
```

- CPU 工作使用 `cpu`；GPU 按硬件选择 `rtx-batch` 或 `b200-batch`。交互调试使用对应 `*-devel`，并设置明确的 `--time`、CPU 和内存。
- 批处理脚本必须显式指定日志路径，例如 `job_logs/<name>_%j.log`；提交后记录 Job ID、脚本、资源、日志和结果路径。
- 取消作业只使用明确的 Job ID：`ssh aicr 'scancel JOBID'`。禁止 `scancel -u $USER` 或取消全部作业。
- 查看日志使用 `tail -f /scratch/jiabenchen_umass/yz/code/job_logs/<name>_<jobid>.log`；长任务不要在 SSH 会话中前台运行。

## 同步与文件安全

- 远程项目目录：`/scratch/jiabenchen_umass/yz/code`。日常传输优先 `rsync`；大文件或大量文件按官方建议使用 Globus。
- 任何删除或覆盖前先做 dry-run：

```bash
rsync -azn --exclude '.git/' --exclude '__pycache__/' ./ aicr:/scratch/jiabenchen_umass/yz/code/
```

- 只有确认预览结果后才执行同步；默认不使用 `--delete`。若确需镜像同步，先确认远程路径和备份，再单独执行。
- 不把数据集、模型权重、密钥或生成视频同步进 Git；日志统一放远程 `job_logs/`，可审阅媒体放本地 `data/<task>/`。

## 本仓库快捷流程

**每次提交远程 job 之前，必须先同步当前本地工作区，再提交。** 不要直接在可能过期的远程 checkout 上 `sbatch`。推荐使用仓库封装脚本，它会先同步再执行 `sbatch`：

```bash
# 同步并提交一个 sbatch
scripts/aicr_submit.sh scripts/<job>.sbatch

# 查看状态与历史
ssh aicr 'squeue -u $USER'
ssh aicr 'sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode'
```

如果不能使用封装脚本，至少按以下顺序操作，并在提交前检查同步预览：

```bash
rsync -azn --exclude '.git/' --exclude '__pycache__/' ./ aicr:/scratch/jiabenchen_umass/yz/code/
rsync -az --exclude '.git/' --exclude '__pycache__/' ./ aicr:/scratch/jiabenchen_umass/yz/code/
ssh aicr 'cd /scratch/jiabenchen_umass/yz/code && sbatch scripts/<job>.sbatch'
```

遇到连接、配额、证书或调度异常，先保留 Job ID 和日志，再联系 AICR 支持；不要通过高频重连、绕过 Slurm 或在登录节点硬跑来规避问题。
