# 本地与 AICR 同步

远程目录：`aicr:/scratch/jiabenchen_umass/yz/code`

本地仓库根目录执行：

```bash
# 本地 → AICR（首次上传或发布本地改动）
rsync -az --delete --exclude '.git/' --exclude '__pycache__/' ./ aicr:/scratch/jiabenchen_umass/yz/code/

# AICR → 本地（先预览，再同步远程改动）
rsync -azn --exclude '.git/' --exclude '__pycache__/' aicr:/scratch/jiabenchen_umass/yz/code/ ./
rsync -az --exclude '.git/' --exclude '__pycache__/' aicr:/scratch/jiabenchen_umass/yz/code/ ./
```

首次同步使用本地 → AICR 命令。`--delete` 会删除远程中本地已不存在的文件，请确认预览结果后再使用。

## 本地修改后直接提交远程作业

仓库提供了封装脚本：它会先同步当前工作区，再在 AICR 上执行 `sbatch`。

```bash
scripts/aicr_submit.sh scripts/render_pour_water_nyx.sbatch
scripts/aicr_submit.sh scripts/finetune_dp3.sbatch --time=1:00:00 -- --task pour_water --sanity
```

查看队列和日志：

```bash
ssh aicr 'squeue -u $USER'
ssh aicr 'tail -f /scratch/jiabenchen_umass/yz/code/job_logs/<job-log>'
```

脚本依赖本机 SSH 配置中的 `aicr` 别名，并假定远程环境已经安装好项目依赖和 Python 环境。
