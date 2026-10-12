# 代码架构与实现说明

本仓库面向华为昇腾（Ascend）NPU 硬件架构，实现了「带复杂布尔属性过滤的高维向量 Top-K 全量召回」高性能检索引擎。`src/engine/` 为检索引擎核心代码，外层配套提供了框架适配层、评测驱动（Harness）、构建流水线以及自动化实验工具。

工程最终编译生成四个可执行文件，产物均输出至 `build/` 目录：

| 可执行文件 | 入口源文件 | 功能与职责 |
|---|---|---|
| `fr_converter` | `src/converter/build_input.cpp` | 将原始数据集 `dataset.bin` 转换为 `fr_builder` 所需的分段输入格式 |
| `fr_builder` | `src/engine/indexer/builder/main.cpp` | 离线索引构建器，生成分片与分段物理索引 |
| `fr_search` | `src/harness/main.cpp` | 检索评测驱动，负责加载索引、分发查询、统计延迟并校验召回率 |
| `fr_npuload` | `src/harness/npu_load.cpp` | 硬件负载扰动工具，用于模拟多卡环境下的算力与显存带宽争用 |

## 1. 目录结构

| 目录路径 | 模块说明 |
|---|---|
| `src/engine/` | 检索引擎核心实现（82 个源码文件），包含索引构建（`indexer/`）、检索执行（`retrieval/`）、索引结构（`index/`）、存储格式（`format/`）及公共定义（`core/`） |
| `src/harness/` | 运行驱动与评测框架，涵盖任务调度、输入输出、召回校验、内存监控及 NUMA 绑核逻辑 |
| `src/platform/` | 平台基础组件桩（Stub），提供日志、异步执行器、配置解析、安全函数库（`securec`）及 MurmurHash |
| `src/device/` | AscendC 设备侧内核代码，包含针对 NPU 硬件加速优化的算子实现 |
| `src/converter/` | 数据预处理工具源码（`fr_converter`） |
| `src/io/` | 跨模块共享的数据 I/O 组件，提供 `dataset_hw.h` 与 `record_io.h` 格式解析 |
| `src/compat/` | 兼容层，提供针对不同 Protobuf 版本的兼容适配头文件 |
| `src/tests/` | 宿主侧单元测试集，采用纯 C++17 实现，不依赖 CANN 环境与 NPU 设备 |
| `run.sh` | 全流程执行驱动脚本，封装编译、查询转换、数据处理、建库及检索评测各阶段 |
| `tools/` | 实验驱动脚本（`exp.sh`）及数据生成与分析脚本 |
| `run/` | 针对核心验收评测项封装的零参数执行脚本 |
| `scripts/` | 依赖检测与安装脚本（`install_deps.sh`）及代码规范化脚本（`format.sh`） |

**模块设计细节**：
- 除 `src/utils/env_switch.h` 外，`src/platform/` 均为外部平台依赖的接口桩。`env_switch.h` 负责统一读取运行时动态配置开关，置于该目录是为了满足 `engine` 模块的头文件包含路径规范。
- 引擎源码中统一使用 `#include "src/full_recall/..."` 路径引用头文件。CMake 构建配置会在构建目录下自动创建指向 `src/engine` 的符号链接（`src/CMakeLists.txt:126`），以保证包含路径与工程结构对齐。

## 2. 编译架构

项目采用设备侧与宿主侧独立的双构建树设计：

| 构建目录 | 源码路径 | 构建产物 | 架构考量 |
|---|---|---|---|
| `build_device/` | `src/device` | `libascend_device.so` 与 `aclrtlaunch_*.h` | AscendC 算子构建流程缺乏幂等性，因此独立构建生成预编译动态库，供宿主侧链接 |
| `build/` | `src/` | 四个宿主侧可执行文件 | 宿主侧主构建树，支持常规的增量编译 |

设备侧实现了四个针对昇腾架构的专用算子内核：
```
src/engine/retrieval/device/filter/kernel_bitmap_text_filter.cpp   # 属性过滤位图计算内核（Vector 单元）
src/engine/retrieval/device/scorer/kernel_vector_mmad_op.cpp       # 向量内积打分内核（Cube 单元）
src/engine/retrieval/device/aggregator/kernel_aggregator.cpp       # 候选结果过滤聚合内核
src/engine/retrieval/device/topk/kernel_topk.cpp                   # 设备侧局部 Top-K 提取内核
```

执行 `run.sh --compile 1` 时，脚本会先清理历史构建目录，依次执行两棵构建树的编译流程，并验证目标可执行文件是否生成。所需编译依赖（CANN、Protobuf、gflags、RapidJSON）均由 `scripts/install_deps.sh` 负责检测。

## 3. 数据流与存储布局

系统数据流分为底库数据处理与查询生成两条主链路，最终汇聚至 `fr_search` 执行检索：

```
dataset.bin (HYDSET2)  --[fr_converter]-->  builder_input/shardN  --[fr_builder]-->  index/shardN
QueryData.txt + tag_map.bin  --[convert_querydata.py]-->  queries.fvecs / filters.txt / topk.txt
```

- **数据集格式**：`dataset.bin` 格式定义于 `src/io/dataset_hw.h`，由 40 字节的文件头、紧随其后的 `docNum × dim` 个 float32 向量数据，以及末尾的 `docNum × ceil(tagNum/64)` 个 uint64 属性标签位图组成。`fr_search` 在 CPU 侧计算召回真值基准时直接通过内存映射（mmap）读取该文件。
- **运行目录布局**：由 `run.sh` 中的 `WORK_DIR` 派生派生（`run.sh:69`）：

| 目录路径 | 存放内容 |
|---|---|
| `$WORK_DIR/queries/` | 转换后的查询集文件（`queries.fvecs`、`filters.txt`、`topk.txt` 及 `manifest.json`） |
| `$WORK_DIR/builder_input/shardN/` | 各分片的 `schema.json` 及分段输入数据，供 `fr_builder` 使用 |
| `$WORK_DIR/index/shardN/` | 最终生成的物理索引目录，供 `fr_search` 载入检索 |
| `$RESULT_DIR/log/` | 编译构建与建库阶段的输出日志 |

查询转换结果与索引剪枝阈值解耦，单次转换即可跨阈值长期复用，`tools/exp.sh` 默认将其存放在 `den0/work/queries` 目录下共享。

## 4. 索引构建流程

索引构建采用两级分块结构：先按硬件卡数切分**分片（Shard）**，再在分片内部按记录数划分为**分段（Segment）**：

- **分片切分**：将总文档数 `DOCS` 均等划分为 `SHARD_NUM` 个互不重叠的分片（`run.sh:218`）。每个分片独立调用 `fr_converter` 与 `fr_builder`，生成一套完整且独立的分片索引。
- **分段切分**：每个分片内部按 `DOC_NUM_PER_SEGMENT` 切分分段。例如一千万总文档切分为 2 个分片、每段包含 131,072 条记录时，单个分片包含 39 个段。

`fr_builder` 针对每个分段依次调用三个构建组件生成分段文件，所有分段完成后调用 `MetaBuilder` 汇总生成全局元数据（`builder_impl.cpp:90`）：

| 构建阶段 | 生成文件 | 内容说明 |
|---|---|---|
| `IDMappingBuilder` | `.idm` | 全局文档 ID（DocID）到段内局部编号的映射表 |
| `VectorBuilder` | `.vector` | 向量连续存储块 |
| `InvertedBuilder` | `.posting` | 属性标签倒排索引列表 |
| `MetaBuilder` | `.meta` | 全局索引元数据信息 |

**并发策略与性能监控**：
- **段级并发**：段间构建并发度受环境变量 `NPUR_EXECUTOR_THREADS` 控制（`executor.h:41`，默认单线程；`run.sh` 默认提升至 `nproc`）。
- **分片级并发**：各分片间默认串行构建；若开启 `BUILD_SHARD_PARALLEL=1`，分片之间将并发处理，此时线程配额在各分片间均分，以避免 CPU 核心资源过度争用（`run.sh:1029`）。
- **耗时输出**：转换与建库耗时分别记录在日志的 `[ConvertTime]` 与 `[BuildTime]` 行，段级详细分解记录于 `[SegTime]` 行。

**倒排压缩与密度阈值**：
倒排索引存储体积由剪枝密度阈值控制。`fr_converter` 将阈值写入 `schema.json`，`fr_builder` 依据该阈值决定属性标签的存储格式（`posting_serializer.cpp:65`）：命中比例低于阈值的标签采用稀疏列表编码，达到或高于阈值的标签采用稠密位图编码。系统支持五种倒排编码布局（`posting_layout.h`），布局标识存储于倒排链头部。调整密度阈值需重新执行数据转换阶段（`--convert-data 1`）。

## 5. 检索执行与调度体系

`fr_search` 的核心检索执行流程定义于 `src/harness/main.cpp` 的 `main()` 函数中：

1. **环境初始化**：调用 `aclInit` 初始化昇腾运行时环境。
2. **设备与分片分配（`ResolveShards`）**：解析 `--shard_index_dirs` 与 `--device_ids`，建立分片索引到物理卡号的映射，并根据 `--card_cpus` 绑定各卡的 NUMA 亲和性。
3. **加载分片（`LoadShards`）**：为每个分片初始化 `DataTable` 结构；开启 `--parallel_load` 时以多线程并发加载各分片。
4. **流池初始化（`InitStreamPool`）**：创建 NPU 异步硬件流池，保障算子间流水并发。
5. **加载查询（`LoadRunInputs`）**：读取向量、过滤条件及 Top-K 配置文件，将布尔过滤表达式解析为抽象语法树（AST，`filter_expr.h`）。
6. **多模式调度**：依据所选模式将查询批次分发至计算设备并驱动检索。
7. **指标统计**：输出各分位数时延统计（P50/P99/Avg），可选输出显存与主机内存快照。
8. **召回校验（`CheckRecall`）**：基于 CPU 真值对比计算召回率。

### 调度模式分类

`src/harness/run/` 提供了四种底层的批次调度实现，由 `main.cpp:174` 根据命令行参数选择执行。`tools/exp.sh` 将其封装为五种测试模式：

| 调度模式 | 硬件资源映射 | 适用场景与测试目标 |
|---|---|---|
| `serial` | 单批次查询在全部分片上并行分发，批次间严格串行 | 用于测量纯服务时延；唯一支持 `--target_qps` 开环定速加压 |
| `groups` | 物理卡划分为多个计算组，每组持有一份全量索引，组间通过工作窃取动态调度 | 兼顾吞吐与单请求时延 |
| `group_pipeline` | 在 `groups` 基础上，组内进一步重叠主机传输与设备计算 | 生产推荐的默认高效模式 |
| `pipeline` | 全局单组，批次 N 的主机提取与批次 N+1 的设备端计算重叠 | 用于极限吞吐评测 |
| `round_robin` | 每张卡独立持有一份全量索引，批次查询轮转分发至单卡处理，无需跨卡合并 | 验收评测中的无通信开销扩展性基线 |

**调度约束说明**：
- 开环加压参数 `--target_qps` 仅在 `serial` 模式中实现（`schedule_serial.cpp`），在分组模式下设置该参数不会生效（分组模式仅考核闭环服务时间）。
- 分组调度模式下的有效文档总数按单个计算组内的分片总和计算（`setup.cpp:130`），分片数量需与组内卡数严格对齐，以避免部分分片遗漏导致全量检索失效。

### 批次内部执行管线

单批次检索在设备侧的处理阶段依次如下（`full_recall_searcher.cpp:532`）：

1. **向量打分（scorer）**：调用 `kernel_vector_mmad` 算子，在 Cube 单元上执行查询向量与文档向量的矩阵乘加打分。
2. **属性过滤（filter）**：调用 `kernel_bitmap_text_filter` 算子，在 Vector 单元上求值布尔表达式并输出命中位图。
3. **聚合与提取（aggregator + topk）**：调用 `kernel_aggregator` 与 `kernel_topk` 算子，在显存内完成相似度与过滤位图的融合筛选，提取局部 Top-K 结果。
4. **结果提取（Extract）**：将设备侧局部 Top-K 传输回主机内存，完成跨分片结果归并与全局排序。

**硬件重叠优化**：
默认启用 `OVERLAP_FILTER` 机制（`full_recall_searcher.cpp:566`），将 filter 阶段的算子准备与执行重叠进 scorer 的矩阵运算窗口中，使 NPU 的 Vector 计算单元与 Cube 计算单元能够并发工作。

**执行接口差异**：
`serial` 模式调用集成的 `BatchSearch` 接口完成单步执行；分组与流水模式则调用 `BatchSearchDevice` 与 `BatchSearchExtract` 两阶段解耦接口，以支持跨批次的双缓冲重叠。

### 召回率校验机制

`CheckRecall` 流程通过内存映射直接读取 `dataset.bin`，在 CPU 侧执行浮点向量暴力搜索计算真实 Top-K，并与引擎返回的文档 ID 进行比对求交。

为避免全量测试时重复计算 CPU 真值造成显著耗时，真值结果会被持久化缓存至 `$RUNS_DIR/recall_ref.bin`。缓存校验指纹涵盖数据集、查询集、过滤条件及 Top-K 请求文件的元数据信息（`recall_ref.cpp:148`），与卡数、分组模式及 Batch Size 无关，因此跨调度模式可完全命中。缓存文件头部包含格式版本号，算法或数据结构变更时会自动失效并触发重新计算。

## 6. 参数与开关分层体系

系统设计了三层参数传递与解析机制，由 `run.sh` 负责转换与下发：

| 配置层级 | 表现形式 | 参数规模 |
|---|---|---|
| 实验工具层 | `tools/exp.sh` 白名单参数（非法输入直接报错拦截） | 50 个 |
| 任务脚本层 | `run.sh` 环境变量（如 `SHARD_GROUP_SIZE`、`STREAM_POOL`） | 70+ 个 |
| 底层引擎层 | `fr_search` gflags 参数（41 个）与 `engine` 内部 `NPUR_*` 环境开关（45 个） | 86 个 |

底层 `NPUR_*` 开关的读取与解析统一位于 `src/platform/src/utils/env_switch.h`。各开关独立指定默认状态，新引入的功能开关默认保持关闭以确保基线稳定，中性开关经性能验证后逐步收敛并移除。

## 7. 测试与实验入口体系

| 入口路径 | 职责与适用场景 |
|---|---|
| `src/tests/run_tests.sh` | 主机侧轻量级单元测试集，涵盖过滤表达式解析、查询与数据集读写、记录序列化及召回真值算法。纯 C++17 实现，无 NPU 设备与驱动依赖 |
| `tools/exp.sh` | 综合实验管理脚本，支持 `latency`、`load`、`index`、`size`、`build` 及 `ref` 六大场景，简化复杂参数的手动配置 |
| `run/*.sh` | 核心验收项的一键执行入口（`load.sh`、`cards.sh`、`index.sh`），内置参数封装，通过退出码反馈测试结果 |
