# 实测数据与踩坑记录

以下数据基于 QAIRT 2.50.0、Qualcomm AI Hub（QCS8550 Proxy）和一块运行 Ubuntu 22.04 的 QCS8550 开发板；
模型是 GPT-SoVITS v2Pro 的一个中文音色（训练数据 19 分钟，GPT e15、SoVITS e16）。

## 板上实测

| 项目 | 结果 |
|---|---|
| 服务启动 | 加载 6 个 context 1.8 s + 预热 2.7 s |
| 12.34 s 语音（3 句）| 合成 5.70 s，RTF 0.46 |
| 流式首段音频（第一句 16 字） | 2.03 s |
| 每个图（板上，含 Python 调度） | g2pw 19.1 ms、bert 17.1 ms、t2s_prefill 65 ms、**t2s_decode 12.4 ms/步**、vits_enc 47.6 ms、vits_gen 97.4 ms/窗 |

耗时大头是 GPT 的逐 token 解码：每秒语音 25 个 token，即 25 × 12.4 ms ≈ 0.31 s。

## AI Hub 真机（fp16，全部算子在 NPU 上）

| 图 | 延迟 | 与 fp32 的误差 |
|---|---|---|
| g2pw_B8_S64（BERT 主体） | 18.2 ms | 1795/1795 个多音字判断与原模型一致 |
| bert22_S128 | 13.9 ms | cos ≥ 0.99997 |
| t2s_prefill_P512 | 36.0 ms | logits 相对误差 ~1.2e-3，top-15 候选 100% 一致 |
| t2s_decode_L1024 | 11.95 ms | 相对误差 ~1.4e-3，top-15 候选 100% 一致 |
| vits_enc_F500_T128 | 47.9 ms | z 相对误差 < 1e-3 |
| vits_gen_W96 | 97.2 ms/窗 | 波形 SNR ~31 dB；log-mel L1 0.058（fp32 模型输出与原始录音之间是 1.08） |

## 为什么这样切图

1. **全部改成静态形状。** GPT-SoVITS 自带的 `onnx_export.py` 只支持 v1/v2，且形状是动态的（KV cache 用 `torch.cat` 增长）。
   这里 KV cache 固定为 L=1024，用加性 mask 标出有效位置；embedding 查表、位置编码、采样、KV 维护放在 CPU。
2. **HiFiGAN 分窗运行。** VITS 整图（10 s）link 失败：`/dec/ups.3` 的 ConvTranspose 需要 27 MB TCM，v73 只有 8 MB。
   把 generator 单独成图，按 96 帧窗口、两侧各 12 帧重叠运行，**首尾窗口贴住序列边界**（不要补零，否则边界 5 帧误差明显），
   拼接结果与整段生成的 SNR 为 106 dB。
3. **BERT 只取前 22 层。** GPT-SoVITS 用的是 `hidden_states[-3]`，即第 22 层的输出，后两层不用算。
4. **参考音色离线计算。** HuBERT、ERes2Net（说话人向量）、ref_enc 只和参考音频有关，在 PC 上算成 `ref_<音色>.npz`，
   设备上不需要这些模型。注意 `ge` 由 SoVITS 权重计算，**换 SoVITS 后必须重新生成**。
5. **mask 用 −100 而不是 −1e4。** 结果完全一样（生成的 token 逐个相同），对 fp16 / 量化的动态范围更友好。

## G2PW（多音字模型）上 NPU 的两个坑

- **输出头不能用 fp16。** 整图 fp16 上真机，1795 行里只有 1720 行的判断和原模型一致（最差 cos 0.52）。原因在输出头：
  masked softmax 写成 `exp(logit − 全局 max) × mask / Σ`，当这个字允许的读音 logit 远小于全局最大值时，fp16 下溢为 0；
  另外词性 `ArgMax` 在两类接近时会翻转，整行 mask 随之改变。解决：NPU 只算到被查询字的 hidden（768 维），
  输出头（两个 Gemm、ArgMax、descriptor 查表、sigmoid、softmax）在 CPU 上用 numpy fp32 计算，之后 1795/1795 一致。
- **AI Hub profile 会用随机输入。** `char_ids`、`position_ids`、`token_type_ids` 是 Gather 的下标，随机值越界后真机报
  `Dma execution failed on the skel side ... err 1100`。在图入口对下标做 Clip（对合法输入无影响）即可。

## AI Hub 使用要点

- `--target_runtime qnn_context_binary` 已经没有了，改用 `submit_compile_and_link_jobs`（先编译成 `qnn_dlc`，再 link）。
- 编译后的 context binary **会重排输入顺序**（例如 decode 变成 `kT_cache, v_cache, mask, x`），并把输出改名为
  `output_0..N`（仍按 ONNX 顺序）。运行库按名字喂输入、按序号取输出。
- onnxslim 会把图的输入输出写进 `value_info`，AI Hub 报 `occur in value_info but also in model IO`，需要删掉。
- 客户端读超时只有约 3 s；上传/下载走 256 MiB 分片、没有总超时，经过代理可能永远卡住：所有调用都包一层带总时限的重试，
  下载改用 16 MiB 分片 + 断点续传。
- 量化任务（`submit_quantize_job`）的校准数据要按模型输入的顺序排列，否则报 `has input X but expected Y`。

## 试过但没有采用的

| 方案 | 结果 |
|---|---|
| w8a16 量化（AI Hub，INT8 权重 / INT16 激活，真实数据校准） | decode 11.95 → 11.57 ms（只快 3%），top-15 一致率降到 93%；vits_gen 反而变慢（97 → 181 ms）；vits_enc 的 int16 Conv 编译失败。**继续用 fp16** |
| KV cache 从 1024 缩到 768 | decode 11.95 → 10.10 ms（−15%），输出与 1024 逐字节相同，但收益太小，保留 1024。decode 不只卡在 KV cache 上，每步还要读约 150 MB 权重 |
| 极短输入前加"。"（原版 GPT-SoVITS 的做法） | 对"好的"有帮助，对"你好"反而变差，没有采用 |

## 音质问题的定位

- **吸气时的"嗡嗡"声 → SoVITS 训练过头。** 呼吸本应是宽带噪声，嗡嗡声是其中出现了一串窄带音调峰。
  用"呼吸段 400–4000 Hz 内高于局部中值 6 dB 以上的能量之和"量化（与听感一致）：e4–e16 为 2–4，
  e20 跳到 11–13（用训练录音的真实 token 重合成、或用 GPT 新生成的 token，两种测法结论相同），底模约 5。
  与 GPT 无关（同一批 token 只换 SoVITS 就有无之别），与"数据里有呼吸声"也无关。`export/eval_sovits_ckpts.py` 可以对任意训练目录做这个比较。
- **高频"电流音" → SoVITS 微调产生的 500 Hz 整数倍音调**（11/13/13.5/14.5/15.7 kHz，HiFiGAN 上采样的周期性伪影）。
  底模没有，e4 起就有且各轮波动（e20 最强 16 dB，e16 约 10 dB）。训练数据 12–16 kHz 能量很低，mel loss 的下限截断使这个频段几乎不受约束。
  窄陷波（`--hf-filter notch`）即可去掉；12 kHz 低通会把 12–16 kHz 的空气感也削掉 20 dB，听起来发闷。
- **发闷的另一半原因在模型本身：** 以 300–2000 Hz 为参考，模型输出在 4–10 kHz 比训练录音低 5–6 dB，原版 PyTorch、底模都一样。
  `--presence-db 4` 可以补偿。
- **分段"断裂" → 段尾被截断。** 每段在最后一个音素后只留 5–20 ms 就接上数字静音。原版 GPT-SoVITS 的拼接也只是逐段补 0.3 s 零静音
  （唯一的"连续性"是 batch>1 时把多段拼在一起送 VITS 解码再切开，api_v2 默认 batch=1 不触发）。这里改为首尾 10/40 ms 淡入淡出，
  并按标点决定停顿（句末 0.3 s、逗号处切开 0.15 s）。
- 用 Gemini（agy）试听打分：能听懂内容（转写正确），但对细微的呼吸噪声判断不稳定，三次重复排序都不一致，不适合做评审。

## 开发板上的坑

- QAIRT 2.50 里只有 `aarch64-oe-linux-gcc11.2` 这套库支持 QCS8550（SoC 编号 66）；gcc9.3 那套报 `Unsupported SoC model 66`。要求 glibc ≥ 2.34。
- 用户不在 `system` 组时打不开 `/dev/adsprpc-smd`，报 `createUnsignedPD not supported` + `openSessionForPriority failed 0x200`。
- **进程退出前必须释放 QNN 资源。** 否则退出时 HTP 对象比 FastRPC 会话活得久，先刷 `undefined m_mutex handle object`，再 segfault
  （x86 模拟器上不会出现）。`QnnBackend` 注册了 `atexit`，CLI/服务也会显式 `close()`。
- 多个 context 用 `REGISTER_MULTI_CONTEXTS` 分组加载（共享 spill-fill）；`createFromBinaryListAsync` + shareResources 在 QCS8550 Linux 上不支持。
  本模型 6 个 context 合计约 1.4 GB，板上一次加载成功。
- 服务里所有 NPU 调用都在同一个线程上执行；请求用 asyncio 锁串行化。不能用线程锁：流式响应跨 `await` 持锁时，单线程池会死锁。
- `curl` 的 `time_starttransfer` 只测到响应头（服务先发 header 再开始合成）；测首段音频请用 `head -c 1` 计时。
- x86 HTP 模拟器（`lib/x86_64-linux-clang/libQnnHtp.so`）能跑单图 context binary，很慢（decode 一步约 4 s），适合检查主机端调用代码。
