# 请把 QAIRT SDK 放在这个目录下

本项目的转换流水线和板端部署包需要高通的 **QAIRT SDK**（Qualcomm AI Runtime，原 Qualcomm AI Engine Direct / QNN SDK）。
受高通许可证约束，SDK **不能随本仓库分发**，需要你自行下载后放到这个 `qairt/` 目录中。

## 1. 下载 SDK

本项目基于 **QAIRT 2.50.0.260828** 构建和验证，直接下载（约 2.6 GB）：

- **https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip**

也可以在 [Qualcomm Software Center](https://softwarecenter.qualcomm.com/)（搜索 "Qualcomm AI Runtime"）或
[产品页面](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) 下载。
使用前请阅读并遵守 SDK 附带的许可协议（`LICENSE.pdf`）。

**版本必须和 Qualcomm AI Hub 编译模型时使用的版本一致**，否则开发板加载 context binary 可能失败。
可以在 AI Hub 编译任务的日志里确认版本：日志中会出现 `/qairt_sdk/default/2.50.0/...` 这样的路径。
AI Hub 升级默认 SDK 后，请下载对应版本（把链接中的两处版本号换成新的完整版本号）。

## 2. 解压到本目录

```bash
cd qairt
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip
unzip -q v2.50.0.260828.zip && rm v2.50.0.260828.zip
```

解压后的目录结构应类似于下面两种之一，`scripts/run_pipeline.sh` 都能识别（上面的压缩包解压后是第二种）：

```
qairt/2.50.0.260828/          ← SDK 根目录：bin/  include/QNN/  lib/  sdk.yaml
qairt/qairt/2.50.0.260828/
```
也可以不放在这里，改用环境变量 `QAIRT_SDK=/path/to/sdk` 指定。

## 3. 用到了 SDK 的哪些部分

- `lib/aarch64-oe-linux-gcc11.2/`：板端主机库 `libQnnHtp.so`、`libQnnSystem.so`、`libQnnHtpV73Stub.so`。
  **QCS8550 在 Linux 上只能用这一套**（gcc9.3 那套会报 `Unsupported SoC model 66`），要求板子 glibc ≥ 2.34（Ubuntu 22.04+）。
- `lib/hexagon-v73/unsigned/`：NPU 侧的 `libQnnHtpV73Skel.so`。
- `include/QNN/`：编译板端库 `libgsv_qnn.so` 用的头文件。
- `bin/aarch64-oe-linux-gcc11.2/qnn-platform-validator`：板端 NPU 访问排错工具。
