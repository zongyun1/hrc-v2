# 历史代码归档

项目主架构为 Lightwheel。此目录保存旧代码和阶段实验，不属于当前运行、依赖安装或默认测试范围。

- `genesis/`：原 Genesis benchmark 的任务、环境、baseline 服务、配置、脚本、测试和文档。
  原根目录说明位于 `genesis/README.md`，旧依赖与 `.gitmodules` 仅作历史记录。
- `experiments/`：早期 Isaac kitchen、RoboCasa 迁移、Lightwheel 初探、人体共仿真和研究记录。
- `manifest.json`：原跟踪文件到归档位置的映射，记录源提交与文件取自工作区或 Git。

已在本地删除的跟踪文件从提交 `238d039be495e66faf7cdfa720cfa37b9a76db30` 恢复到归档。
现存工作区文件优先保留。资产、外部仓库、模型与产物仍独立管理，不复制进 Git。
历史脚本保留原路径和依赖假设，不能从新的项目根目录直接作为当前任务入口运行。
需要复现实验时应使用独立工作区及对应环境；原 Genesis 完整 Git 布局可从上述提交查看。
当前代码不得导入 `history` 中的模块。
