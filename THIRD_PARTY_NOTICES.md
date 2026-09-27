# 第三方代码与许可

| 路径 | 来源 | 许可证 |
|---|---|---|
| `device/text/` | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（v2Pro，20250604）的中文文本前端：`chinese2.py`、`tone_sandhi.py`、`cleaner.py`、`symbols*.py`、`opencpop-strict.txt`、`g2pw/`、`zh_normalization/`。已修改：去掉 torch / onnxruntime 的硬依赖、G2PW 会话可替换（改由 NPU 执行）、jieba_fast 缺失时回退到 jieba、模型路径改为相对本目录 | MIT，见 `device/text/LICENSE.GPT-SoVITS` |
| `device/text/g2pw/`、`device/text/zh_normalization/` | GPT-SoVITS 中该部分又来自 [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) 与 [g2pW](https://github.com/GitYCC/g2pW) | Apache-2.0 |
| `device/src/gsv_qnn.cpp` | 在 [qwen3-reranker-qnn](https://github.com/zhouzengming/qwen3-reranker-qnn) 的 `qnn_reranker.cpp` 基础上改写 | Apache-2.0 |
| `export/aihub_download.py` | 改写自 qwen3-reranker-qnn 的同名脚本 | Apache-2.0 |

本仓库**不包含**、也不分发以下内容，需要用户自行获取：
- 高通 QAIRT SDK（QNN 运行库、头文件、工具）：受高通许可证约束，见 `qairt/README.md`。部署包中的 `qnn_libs/`、`include/`、`tools/` 由 `export/make_deploy.py` 从用户本地的 SDK 拷贝。
- GPT-SoVITS 的代码与预训练模型（chinese-roberta-wwm-ext-large、chinese-hubert-base、ERes2NetV2、G2PWModel 等），以及用户自己训练的模型、训练数据和参考音频。
