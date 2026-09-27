# 请把 QAIRT SDK 放在这个目录下

本项目的转换流水线和板端部署包需要高通的 **QAIRT SDK**（Qualcomm AI Runtime，原 Qualcomm AI Engine Direct / QNN SDK）。
受高通许可证约束，SDK **不能随本仓库分发**，需要你自行下载后放到这个 `qairt/` 目录中。

## 1. 下载 SDK

- 下载地址：[Qualcomm AI Runtime SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
  （需要登录高通开发者账号，并同意许可协议）。
- **版本必须和 Qualcomm AI Hub 编译模型时使用的版本一致。** 本项目基于 **2.50.0**（2.50.0.260828）构建和验证。
  版本不一致时，开发板加载 context binary 可能失败。可以在 AI Hub 编译任务的日志里确认版本：日志中会出现
  `/qairt_sdk/default/2.50.0/...` 这样的路径。

## 2. 解压到本目录

把压缩包直接解压到这里。解压后的目录结构应类似于下面两种之一，`scripts/run_pipeline.sh` 都能识别：

```
qairt/
├── README.md                 ← 本文件
└── 2.50.0.260828/            ← SDK 根目录（名称随版本号变化）
    ├── bin/  include/QNN/  lib/{aarch64-oe-linux-gcc11.2,hexagon-v73,...}/  sdk.yaml
```
或者多一层 `qairt/qairt/2.50.0.260828/`。也可以不放在这里，改用环境变量 `QAIRT_SDK=/path/to/sdk` 指定。

## 3. 用到了 SDK 的哪些部分

- `lib/aarch64-oe-linux-gcc11.2/`：板端主机库 `libQnnHtp.so`、`libQnnSystem.so`、`libQnnHtpV73Stub.so`。
  **QCS8550 在 Linux 上只能用这一套**（gcc9.3 那套会报 `Unsupported SoC model 66`），要求板子 glibc ≥ 2.34（Ubuntu 22.04+）。
- `lib/hexagon-v73/unsigned/`：NPU 侧的 `libQnnHtpV73Skel.so`。
- `include/QNN/`：编译板端库 `libgsv_qnn.so` 用的头文件。
- `bin/aarch64-oe-linux-gcc11.2/qnn-platform-validator`：板端 NPU 访问排错工具。
