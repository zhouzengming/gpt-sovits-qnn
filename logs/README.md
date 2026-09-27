# 请把 GPT-SoVITS 的训练目录复制到这里

在 GPT-SoVITS 中完成训练后，把 `GPT-SoVITS/logs/<实验名>/` **整个目录**复制到本目录下，例如 `logs/myvoice/`，
然后运行 `GSV_ROOT=/path/to/GPT-SoVITS bash scripts/run_pipeline.sh myvoice`。

转换脚本会用到其中的：

| 路径 | 用途 |
|---|---|
| `logs_s1_v2Pro/ckpt/epoch=N-step=M.ckpt` | GPT 模型（`epoch=N` 对应 GPT-SoVITS 导出的 `-e<N+1>`），用 `GPT_EPOCH` 选择，默认最后一轮 |
| `logs_s2_v2Pro/G_<step>.pth` + `config.json` | SoVITS 模型（按文件内记录的轮数编号），用 `SOVITS_EPOCH` 选择，默认最后一轮 |
| `5-wav32k/`、`2-name2text.txt` | 从训练音频中自动挑选一条 3~10 秒的参考音频及其文本（用 `REF=<文件名>` 指定） |
| `6-name2semantic.tsv`、`7-sv_cn/` | `export/eval_sovits_ckpts.py` 评估各轮 SoVITS 用 |

只支持 **v2Pro / v2ProPlus**、**中文**模型。本目录下除本文件外的内容不会被提交到 git。

> 小数据集上 SoVITS 训得越久不一定越好：实测一个 19 分钟的音色，最后一轮（e20）在吸气处有明显的"嗡嗡"声，
> e8–e16 没有。转换前可以先运行 `python export/eval_sovits_ckpts.py --exp logs/<实验名>` 比较各轮。
