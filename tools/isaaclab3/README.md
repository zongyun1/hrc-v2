# Isaac Lab 3 开发工具

原生源码位于项目根目录，适配已经合入。这里不再保存独立 source checkout。

- `setup.sh`：向现有 Lab 3 容器安装根目录项目和 vendored Arena。
- `assets.py`：从已下载的上游 checkout 复制并校验 LFS 资产。
- `migrate.py`：可重复执行的适配工具，目标为项目根目录；修改源码后无需每次运行。
- `robot_materials.py`：Panda 白色材质恢复。
- `run_human_navigation.py` / `run_human.sbatch`：固定场景避人导航示例。
- `smoke.py` / `run.sbatch`：OpenDrawer 相机与接口检查。
- `test_*.py`：CPU 回归。

从项目根目录运行 `bash install.sh` 或 `sbatch tools/isaaclab3/run_human.sbatch`。
默认 benchmark 入口为 `python -m hrc_bench`。
历史验证见 [迁移记录](../../docs/isaaclab3_migration_validation.md)。
