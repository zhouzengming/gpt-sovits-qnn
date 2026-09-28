# GPT-SoVITS 高通 NPU 部署（QNN / HTP）

把 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) **v2Pro** 训练出的中文音色转换到高通 **QCS8550** 开发板
（Ubuntu，aarch64）的 Hexagon NPU 上运行：6 个模型全部以 fp16 在 NPU 上执行，CPU 侧只用 numpy。
提供命令行推理和 **OpenAI 兼容的 TTS 服务**（`/v1/audio/speech`）。

| 在 QCS8550 开发板上（12.34 s 语音，3 句） | |
|---|---|
| 合成耗时 | **5.70 s（RTF 0.46）** |
| 流式首段音频 | **2.03 s** |
| 服务启动（加载 + 预热） | 4.5 s |
| 清晰度（whisper-small 转写 CER） | 与原版 GPT-SoVITS PyTorch 推理持平 |

## 原理

```
text ─(CPU: 分句, jieba, 规则)+[g2pw 多音字]─> phones ─[bert]─> 音素级 BERT 特征
     ─[t2s_prefill] + N×[t2s_decode] (CPU: 采样、KV cache)─> semantic tokens
     ─[vits_enc] + [vits_gen]×窗口─> 32 kHz wav
```

`[...]` 是 NPU 上的静态形状图（QNN context binary，fp16）：

| 图 | 内容 | 形状 |
|---|---|---|
| `g2pw` | 多音字模型的 BERT 主体（输出头在 CPU 上用 fp32 算，fp16 会判错读音） | 8 个多音字 × 64 token |
| `bert` | chinese-roberta-wwm-ext-large 前 22 层（= GPT-SoVITS 用的 `hidden_states[-3]`） | 128 token |
| `t2s_prefill` / `t2s_decode` | GPT（text-to-semantic），KV cache 固定长度 + mask | prefill 512，KV 1024 |
| `vits_enc` | SoVITS 编码器 + flow | 500 帧（10 s）、128 个音素 |
| `vits_gen` | HiFiGAN，按 96 帧窗口、重叠 12 帧运行（整段需要 27 MB TCM，v73 只有 8 MB） | 96 帧 |

参考音色（HuBERT 语义 token、说话人向量、音色向量）在 PC 上预先算好，设备上不需要这些模型。
设计取舍、实测数据和踩过的坑见 [docs/NOTES.md](docs/NOTES.md)。

## 目录结构

```
logs/            把 GPT-SoVITS 的训练目录 logs/<实验名>/ 复制到这里（见 logs/README.md）
qairt/           把高通 QAIRT SDK 解压到这里（见 qairt/README.md）
export/          主机端转换流水线
  export_onnx.py      1. 导出 6 个静态 ONNX 图（+ G2PW 输出头参数）
  verify_onnx.py      2. 与 GPT-SoVITS 原实现逐图对比
  prepare_assets.py   3. CPU 侧权重、参考音色、分词器
  aihub_compile.py    4. AI Hub 编译 fp16 context binary 并测速（可断点续跑）
  aihub_download.py   5. 分片下载 context binary
  make_deploy.py      6. 组装板端部署包（QNN 运行库从本地 SDK 拷贝，可交叉编译板端库）
  aihub_verify.py     （可选）在 AI Hub 真机上验证 context binary
  eval_sovits_ckpts.py（可选）比较各轮 SoVITS 的呼吸"嗡嗡"声和高频细音，帮助选轮次
scripts/run_pipeline.sh   一键执行 1–6
device/          板端代码（部署包的内容）
  gsv_runtime.py      推理运行时 + 命令行
  serve_tts.py        OpenAI 兼容 TTS 服务
  src/gsv_qnn.cpp     QNN C API 运行库（ctypes 调用）
  text/               GPT-SoVITS 中文前端（去掉了 torch / onnxruntime 依赖）
docs/NOTES.md    实测数据与踩坑记录
```

## 环境要求

- **主机**：x86_64 Linux，Python 3.10+，`pip install -r requirements-export.txt`（torch 用 CPU 版即可，见文件内说明）。
  建议内存 16 GB 以上（在 26 GB 内存的机器上验证）。
- **GPT-SoVITS**：你训练用的 GPT-SoVITS 目录（用到其中的模型代码和 `GPT_SoVITS/pretrained_models` 下的预训练模型：
  chinese-roberta-wwm-ext-large、chinese-hubert-base、sv、以及 `GPT_SoVITS/text/G2PWModel`）。用环境变量 `GSV_ROOT` 指定。
  在 v2Pro 整合包（20250604）上验证。
- **Qualcomm AI Hub 账号**：`pip install qai-hub` 后执行 `qai-hub configure --api_token <token>`。
- **QAIRT SDK 2.50.0**：解压到 `qairt/`（见 [qairt/README.md](qairt/README.md)），版本需与 AI Hub 编译时一致。
- **开发板**：QCS8550，Ubuntu 22.04+（glibc ≥ 2.34），运行用户在 `system` 组中。

## 转换自己的音色

```bash
# 1. 把 GPT-SoVITS 训练得到的 logs/<实验名>/ 整个复制到本仓库的 logs/ 下
cp -r /path/to/GPT-SoVITS/logs/myvoice logs/

# 2. 把 QAIRT SDK 解压到 qairt/（见 qairt/README.md）

# 3. 运行转换（默认转换最后一轮；可用 GPT_EPOCH / SOVITS_EPOCH 指定轮次，REF 指定参考音频）
GSV_ROOT=/path/to/GPT-SoVITS GPT_EPOCH=15 SOVITS_EPOCH=16 bash scripts/run_pipeline.sh myvoice
```
得到的部署包在 `dist/myvoice/`。AI Hub 编译 6 个图约需 30–60 分钟，中断后重跑会接着之前的任务继续。

**选择轮次。** 小数据集上 SoVITS 训得越久不一定越好（实测一个音色最后一轮在吸气处有明显的"嗡嗡"声，前几轮没有）。
转换前可以比较各轮：
```bash
GSV_ROOT=/path/to/GPT-SoVITS python export/eval_sovits_ckpts.py --exp logs/myvoice
```
两项指标都越低越好；最终以试听为准。参考音频默认从训练音频里自动挑一条 3–10 秒（接近 6 秒）的，也可以用
`REF=<5-wav32k 下的文件名>` 指定，或直接调用 `export/prepare_assets.py --ref-wav ... --ref-text ...` 使用任意音频；
多次调用 `prepare_assets.py --voice <名字>` 可以生成多个音色。

## 在开发板上运行

把 `dist/<实验名>/` 拷到板子上：
```bash
pip install -r requirements-device.txt
source setup_env.sh
python gsv_runtime.py --backend qnn --text "今天天气不错，我们一起去公园散步吧！" --out out.wav
python serve_tts.py --host 0.0.0.0 --port 8000          # OpenAI 兼容服务
```
```bash
curl http://<板子IP>:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"gpt-sovits","input":"今天天气不错。","response_format":"wav"}' -o out.wav
```
接口参数、分段规则、音质选项、systemd 配置和排错见 [device/README.md](device/README.md)。

## 限制

- 只支持 GPT-SoVITS **v2Pro / v2ProPlus**（v2ProPlus 未实测）和**中文**文本。
- 每段最长 10 s（长文本按自然语句切分后，把相邻句子打包成不超过 30 字的段，超长时自动再对半切）；`speed` 只支持 1.0。
- 只在 QCS8550 上验证过；其他 v73 NPU 的 SoC 可能只需改 `DEVICE` / `QNN_TARGET`，未测试。

## 许可证

本仓库代码以 [Apache-2.0](LICENSE) 发布。`device/text/` 来自 GPT-SoVITS（MIT），其中部分又来自 PaddleSpeech / g2pW（Apache-2.0），
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库不包含高通 QAIRT SDK、GPT-SoVITS 预训练模型，也不包含任何训练好的音色。
使用他人声音训练和发布模型时，请确保已获得授权。
