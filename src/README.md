# port/ — 让 `new/`（TianjiEngine full-recall）跑起来的移植层

本目录是**路线 A** 的全部新增工作，与 `old/`、`new/` 平级隔离。`new/` 源码保持不动，
本目录通过 CMake 把 `new/` 的 build/data/codec/search 编起来，用基建桩替换仓库外依赖，
绕过 `process/`（在线服务 + `vector_recall` 耦合层），再用独立 harness 驱动 `FullRecallSearcher`。

整体计划见仓库根的 [TODO.md](../TODO.md)。

## 目录

```
port/
  stubs/            仓库外依赖的最小本地实现（基建桩）
    src/utils/        logger, performance_recorder, file_manager, string_util, alarm
    src/common/       error_code, uncopyable, tianjiengine_constants
    src/configuration/ develop_configuration (.h DECLARE / .cpp DEFINE，gflags)
    src/workflow/async/ executor（同步版，先跑通再并行）
    configuration/    develop_configuration.h 转发头（无 src 前缀的 include）
    murmurhash/       MurmurHash3（common/hash 依赖，公有领域实现）
    huawei_platform/huawei_secure_c/include/securec.h  memcpy_s 等安全函数 shim
  converter/        Phase 5 数据转换器（dataset_HW.bin → builder proto 输入）
  harness/          Phase 6 驱动 main（绕过 process/）
  cmake/            CMake 辅助
  CMakeLists.txt    顶层构建（待写）
```

## include-root 映射（关键）

`new/` 混用三种 include 前缀，构建时用两个 `-I` 根解决：

| include 形态 | 解析到 |
|---|---|
| `"src/full_recall/..."` | 由 CMake 在 build 目录建符号链接 `src/full_recall -> <repo>/new` |
| `"full_recall/..."` | 符号链接 `full_recall -> <repo>/new` |
| `"posting_data.h"`（同目录） | 各源文件所在目录自动解析 |
| `"src/utils/..."` `"src/common/..."` `"configuration/..."` `"murmurhash/..."` `"huawei_platform/..."` | `-I port/stubs` |

## 已完成（Phase 2 基建桩）

以下桩已按 `new/` 真实用法核对签名，并在本地用 Apple clang 对 `new/` 纯 host 切片验证通过：

- 本地编译通过的引擎源：`core/number/trans_number.cpp`、`core/hash/hash.cpp`、
  `format/file_header.cpp`、`format/posting_serializer.cpp`（+ 本目录 `murmurhash/MurmurHash3.cpp`）。
- 验证命令示例：
  ```bash
  c++ -std=c++17 -c new/common/number/trans_number.cpp -Iport/stubs -Inew/common/number
  c++ -std=c++17 -c new/common/hash/hash.cpp          -Iport/stubs -Inew/common/hash
  ```

> `develop_configuration` 依赖 gflags，本地未安装，未做本地编译验证；逻辑为标准 gflags
> DECLARE/DEFINE 拆分，服务器上编译。

## 已完成（Phase 1 / 5 / 6）

- 顶层 [CMakeLists.txt](CMakeLists.txt)：CANN 探测、include-root 符号链接、protoc 生成、
  三个 host 库（`frnew_common`/`frnew_search`/`frnew_build`）、可执行 `fr_builder`/`fr_search`/`fr_converter`。
- Phase 5 转换器 [converter/build_input.cpp](converter/build_input.cpp)：`synthetic` / `hw` 两种数据源
  → `schema.json` + 分段输入文件（本地 fake-proto syntax-check 通过）。
- Phase 6 harness [harness/main.cpp](harness/main.cpp)：绕过 process/ 直驱 `FullRecallSearcher`。

## 端到端运行（NPU 服务器）

```bash
# 0a. 一次性预编译设备 kernel 库（只在 kernel .cpp 变动时重做）。
#     ascendc 的 ExternalProject 构建步非幂等，放在独立 build 树里编一次，
#     主构建直接 import，之后改 host 代码不再触发设备重编、无需 rm -rf。
cmake -S port/device -B build_device \
      -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
      -DSOC_VERSION=Ascend910B3
cmake --build build_device        # 串行！ascendc 的 ExternalProject 子构建在满 -j 下有竞态，
                                  # 会在 merge 步偶发失败；设备库很小，串行即可
# 产物：build_device/lib/libascend_device.so + build_device/include/ascend_device/*.h

# 0b. 主构建（import 预编译设备库；host 迭代可反复 `cmake --build build -j`，很快）
cmake -S port -B build \
      -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
      -DSOC_VERSION=Ascend910B3
cmake --build build -j
# 若未先做 0a，主构建会 fallback 到 inline 编设备库（能编，但之后每次重编都要 rm -rf build）

# 1. 造 builder 输入（先用合成数据打通链路；P 必须是 16 的倍数）
./build/fr_converter --mode synthetic --out /tmp/fr_in --docs 10000 \
      --dim 64 --tag_num 35840 --tags_per_doc 8 --doc_num_per_segment 1024
#   或真实子集： --mode hw --dataset dataset_HW.bin --docs 100000

# 2. 建索引（读 <data_dir>/schema.json）
./build/fr_builder --data_dir /tmp/fr_in --data_out /tmp/fr_out

# 3. 检索（match-all 或 --filter_tag 走过滤路径）
./build/fr_search --index_dir /tmp/fr_out --query_file datasets/hw_queries.fvecs \
      --topk 100 --num_queries 20 --filter_tag -1
```

## 待办

- **Phase 4 编译连通探针**（服务器）：跑上面 step 0，按缺失符号回填桩 / 标记抽取缺文件。
- 惰性补齐 utils 桩：`random_number.h` `time_util.h` `threading/thread_pool.h`
  `single_thread_task.h`（多为 process/ 或次要路径，编译报错时再补）。
- Phase 7 召回验证、Phase 8 性能测量。

## 已知待服务器验证的假设

- `bthread_setconcurrency` 用同步桩（`stubs/.../executor.h`）——绕过 brpc/bthread。
- protobuf 仅消息、无 gRPC service；`ascend_device` 生成的 `aclrtlaunch_*.h` 前缀为 `ascend_device/`。
- `split_doc_num_zn` 默认 = `doc_num_per_segment`、各 `*_block_dim`/topk 阈值默认值，均需实机调。
- libstdc++ ABI（`FRNEW_GLIBCXX_ABI`，默认 1）需与 CANN/protobuf 一致，冲突则改 0。
