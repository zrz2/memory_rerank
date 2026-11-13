# 当前统一实验环境

配置日期：2026-11-13。本文件描述现在的默认配置；更早阶段的 E5 结果保留原始模型身份，不与 BGE-M3 结果混用。

## Git 身份

仓库级 user.name 为 `zrz2`，user.email 为 `m13437876693_1@163.com`。使用已有身份并写入本仓库 local config，不修改其他项目的全局配置。

这是提交署名，不是 GitHub 登录。目前没有远程仓库；本地提交不需要 GitHub token。

## LLM：统一 DeepSeek

- 官方入口：`https://api.deepseek.com`。
- 请求模型：`deepseek-chat`，用户已选择。
- 本次真实联通检查返回 `deepseek-flash`；保留请求模型名与响应模型名，不声称该别名永远指向相同底层版本。
- temperature=0，默认输出上限512 tokens、超时90秒；具体实验如需覆盖应记录配置。
- 规划统一用于 importance、LLM reranking、回答生成及模型评判。当前已完成通用调用客户端和联通验证；这些实验模块的专用提示和缓存仍待后续实现。
- 不读取 OPENAI_API_KEY / OPENAI_BASE_URL，不静默切换服务商。
- 用同一 DeepSeek 生成并评判答案可能存在自偏好。正式实验应做人工抽检，且不能把换用 DeepSeek judge 的成绩直接称为原论文官方 GPT judge 成绩。

密钥位于本项目 `.env` 中的 `DEEPSEEK_API_KEY`，也可通过同名进程环境变量配置，环境变量优先。真实 `.env` 被 Git 忽略；只提交空白 `.env.example`。日志和检查报告只记录密钥是否存在，不记录密钥值。

## Embedding：BGE-M3

- 模型 `BAAI/bge-m3`，MIT，固定 revision `5617a9f61b028005a4858fdac845db406aefb181`。
- 本地目录 `.cache/models/bge-m3`，校验清单 `configs/embedding_files_manifest.json`。
- 当前只使用 dense 向量，1024维、L2归一化；不宣称启用了 sparse 或 ColBERT 三路检索。
- 使用模型自带 CLS pooling；query/passages 均不添加 E5 的 query/passage 前缀。
- CPU，batch_size=2，当前运行上限512 tokens。这是适配本机的运行限制，BGE-M3 原生上下文支持更长；正式长文本仍按明确的切块协议处理。
- 暂不安装 GPU 版 PyTorch，不保证 4GB GPU 能容纳所有输入配置。CPU路径先获得可复现结果。
- 向量缓存键包含完整模型配置和文本，旧 E5 缓存不会与 BGE-M3 混淆。

## 检查命令

在项目目录执行：

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/check_environment.py --embedding
# 以下命令真实调用一次 DeepSeek，产生少量 token 用量：
./.venv/Scripts/python.exe scripts/check_environment.py --embedding --live
git config --local --get user.name
git config --local --get user.email
git check-ignore .env
```

检查报告：`reports/environment_check.json`。下载脚本支持续传，最后以源端 SHA256 核验，未完成校验的文件不进入模型加载。

## 本阶段边界

环境检查只证明模型可加载、接口可调用；不代表论文方法已复现或评测成绩已产生。正式比较时，各方法必须使用相同的 BGE-M3 版本、候选记忆、切块规则、DeepSeek 配置与测试集划分，只改变要研究的排序机制。早期 E5 结果不能与新 BGE-M3 结果直接用于证明重排序收益。

## 2026-11-13 实测验收

- BGE-M3 权重和 tokenizer 已完整下载，文件清单已更新；权重通过源端 LFS SHA256 校验。
- 实际编码 3 条中英文记忆和 1 条中文问题，输出维度1024，向量范数约1，4条均未截断；纯编码耗时约0.50秒（不含加载时间，不能代表长文本吞吐）。
- 中文地址、同义英文地址、无关音乐文本对中文寄件问题的相似度依次为0.6752、0.6436、0.3941。这只是连通和编码检查，不是检索质量评测集。
- DeepSeek 返回 OK，单次请求10 tokens，请求模型 deepseek-chat，响应模型 deepseek-flash；结果已写入 reports/environment_check.json。
- 20项单元测试通过，pip check 无依赖冲突。
