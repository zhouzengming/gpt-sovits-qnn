# GPT-SoVITS 板端部署包（QCS8550 NPU）

```
text ─(CPU: 分句, jieba, 规则)+[g2pw 多音字]─> phones ─[bert]─> 音素级 BERT 特征
     ─[t2s_prefill] + N×[t2s_decode] (CPU: 采样 top_k=15 / 重复惩罚 1.35 / KV cache)─> semantic tokens
     ─[vits_enc] + [vits_gen]×窗口 (96 帧窗口、重叠 12 帧拼接)─> 32 kHz wav
```
`[...]` 是 NPU 上的静态图（`models/*.bin`，QNN context binary，fp16 计算，输入输出为 float32/int32），
通过 `lib/libgsv_qnn.so`（QNN C API，ctypes 调用）执行；其余在 CPU 上用 numpy 完成，不需要 torch 和 onnxruntime。

## 板子要求
- QCS8550（其他 v73 NPU 的 SoC 未验证），Ubuntu 22.04 及以上（本包的 QNN 库为 `aarch64-oe-linux-gcc11.2`，要求 glibc ≥ 2.34）。
- 运行用户需要能访问 `/dev/adsprpc-smd`（通常是 `system` 组）：`sudo usermod -aG system $USER` 后重新登录。
- Python 3.10+。

## 安装与运行
```bash
sha256sum -c --quiet SHA256SUMS          # 校验拷贝是否完整
pip install -r requirements-device.txt
source setup_env.sh                      # 设置 LD_LIBRARY_PATH / ADSP_LIBRARY_PATH
[ -f lib/libgsv_qnn.so ] || make         # 部署包里没有预编译的库时，在板上编译（需要 g++）

python gsv_runtime.py --backend qnn --text "今天天气不错，我们一起去公园散步吧！" --out out.wav
```
命令行参数：`--voice`（`assets/ref_<音色>.npz`）、`--seed`、`--top-k`、`--temperature`、`--hf-filter`、`--presence-db`。

## OpenAI 兼容 TTS 服务（`serve_tts.py`）
```bash
source setup_env.sh
python serve_tts.py --host 0.0.0.0 --port 8000            # 可选 --api-key KEY（要求 Authorization: Bearer KEY）
```
启动时一次性加载 6 个 NPU 模型、CPU 查表权重和 `assets/ref_*.npz` 中的全部音色，并做一次预热合成
（触发 jieba 词典、G2PW 等所有懒加载），之后的请求不再读硬盘。只有一个 NPU，请求按到达顺序串行处理。

| 接口 | 说明 |
|---|---|
| `POST /v1/audio/speech` | OpenAI 格式：`input`（≤4096 字）、`voice`（音色名；OpenAI 预设名如 `alloy` 映射到默认音色）、`response_format`（`wav` / `pcm` / `mp3` / `opus` / `aac` / `flac`，默认 `mp3` 与 OpenAI 一致；后四种需要 `apt install ffmpeg`）、`speed`（只支持 1.0）。扩展字段：`stream`（逐句返回，`pcm` 默认开启）、`seed`、`top_k`、`temperature`、`hf_filter`、`presence_db` |
| `GET /v1/models`、`GET /v1/audio/voices`、`GET /health` | 模型名 / 音色列表 / 是否就绪 |

`pcm` 为 24 kHz 16-bit 单声道（与 OpenAI 相同），其他格式为 32 kHz。

```bash
curl http://<板子IP>:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"gpt-sovits","input":"今天天气不错。","response_format":"wav"}' -o out.wav
```
```python
from openai import OpenAI
cli = OpenAI(base_url="http://<板子IP>:8000/v1", api_key="none")
cli.audio.speech.create(model="gpt-sovits", voice="<音色>", input="今天天气不错。", response_format="wav").write_to_file("out.wav")
```

### systemd 示例
```ini
# /etc/systemd/system/gsv-tts.service
[Unit]
Description=GPT-SoVITS TTS on QCS8550 NPU
After=network.target

[Service]
User=<用户>                     # 该用户需要在 system 组中
WorkingDirectory=/path/to/package
ExecStart=/bin/bash -c 'source setup_env.sh && exec python3 serve_tts.py --host 0.0.0.0 --port 8000'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## 分段与音质
- 长文本按句末标点（。！？!?… 和换行）切句；超过 30 字的句子再按逗号/分号切开（没有标点才硬切）；
  不足 5 字的片段并入相邻片段（合并后不超过 30 字）。字数按文本规范化后的汉字计（"2026年"算 5 个字，标点不计）。
- 每段最长 10 s（250 个 semantic token，由 VITS 图的长度决定）。
- 段与段之间：每段首尾 10 ms / 40 ms 淡入淡出，句末停顿 0.3 s，长句在逗号处切开的地方停顿 0.15 s（服务的 `--pause` / `--clause-pause`）。
- `--hf-filter notch`：只陷波 11/13/13.5/14.5/15.7 kHz，去掉 SoVITS 微调可能带来的高频细音；`lowpass` 为 12 kHz 低通（会发闷）；默认 `off`。
- `--presence-db 4`：约 4 kHz 以上的高架提升（模型输出在 4–10 kHz 比真实录音少 5–6 dB，原版 GPT-SoVITS 也如此）。
- 两三个字的极短输入（如"好的。"）模型本身容易读得含糊，原版同样如此。

## 排错
| 现象 | 原因 / 解决 |
|---|---|
| `Dsp startup: Unsupported SoC model (SnapdragonModel): 66` → `deviceCreate failed` | QNN 库版本不对：QCS8550 在 Linux 上只能用 `aarch64-oe-linux-gcc11.2` |
| `GLIBC_2.34 not found` / `GLIBCXX_3.4.29 not found` | 需要 Ubuntu 22.04 及以上 |
| `createUnsignedPD ... not supported`，接着 `openSessionForPriority failed ... 0x200` | 没有 `/dev/adsprpc-smd` 权限：把用户加入 `system` 组；`bash tools/diagnose_npu.sh` 可确认 |
| `Failed to find available PD ... context size estimate` | HTP 内存不足（6 个 context 合计约 1.4 GB）；运行库会先按 `REGISTER_MULTI_CONTEXTS` 分组加载 |
| `contextCreateFromBinary ... failed`（其他错误码） | QNN 运行库必须与 AI Hub 编译时的 SDK 版本一致 |
| `dlopen ... libcdsprpc.so` 失败 | BSP 缺少 FastRPC 用户态库，或不在 `LD_LIBRARY_PATH` 中 |
