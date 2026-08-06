# TODO — 让 `new/`（TianjiEngine full-recall）在数据集上跑起来（路线 A）

> **【已归档 · 2026-07-04】** 本文是路线 A 的原始规划/TODO,已全部完成并超额:`new/` 端到端跑通、CPU 自校验召回 **100%**、通过调参(段 65536 + block_dim 16)把 1M 的 p99 从 47ms 压到 **~2ms**,并根因修复了 bitlist bug。10M 全量建索引受**容器内存(238GB)/ UB / HBM** 三重工程约束。下方复选框保留当时状态,未逐一回填;保留此文仅作历史记录。

> 路线 A：**真跑 `new/` 代码本身**（不做 old/test/full_npu 的再实现）。
> 核心策略：用 CMake 复用 `old/` 已验证的 CANN/AscendC 工具链；**绕过 `process/`**（在线服务 + `vector_recall` 耦合层），补一层轻量**基建桩**，把 `build/data/codec/search` 编起来，写一个独立 `main` 驱动 `FullRecallSearcher`，用**转换后的数据集**端到端跑通并验召回。

标记说明：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成。风险/位置标签：`(低风险)` `(中风险)` `(需实机)` = 需 NPU 服务器实机验证 · `(本地)` = 本地开发机可做 · `(服务器)` = 仅 NPU 服务器可做。

---

## 背景速记（为什么现在跑不起来）

1. **构建系统**：`new/` 用 Bazel（各目录 `BUILD`），依赖 `@ascend`、`//third_party`、`//huawei_platform`、`//src/utils` 等外部 target；本仓库无 Bazel workspace。
2. **仓库外依赖**：`new/` 是大工程 `src/full_recall/` 子树的抽取快照，`#include` 引用了一批不在 `new/` 内的模块（见下方「外部依赖清单」）。
3. **数据格式**：builder 吃 protobuf `Document`（`Section`=属性 TermInfos + FloatEmbedding 向量）+ `schema.json` + `IndexBuilderConfig`；与 `old/` 的 `dataset_HW.bin` 不兼容。

**利好（已核实）**：
- `old/` 已用 CMake 成功编译 AscendC kernel 并链接 `ascendcl`（[old/CMakeLists.txt](old/CMakeLists.txt)）——最硬的工具链已趟通。
- `new/search/device/CMakeLists.txt` **已存在**，直接把 4 个 kernel 编成 `ascend_device.so`；kernel 只依赖 `kernel_operator.h`(CANN)+本地头，**完全自包含**。
- `vector_recall`/`IndexManager` 耦合**只在 `process/`**，绕过后消失。
- 绕过 `process/` 后，host 侧外部依赖收敛到 **~12 个头**，以 `logger.h` 为主，均可桩化。

---

## Phase 0 — 准备与决策　(本地)

- [ ] **P0.1** 在 NPU 服务器上确认 CANN 版本、`SOC_VERSION`（如 `Ascend910B3`）、`ascendc_kernel_cmake` 路径（`compiler/tikcpp/` 或 `tools/tikcpp/`）。
- [ ] **P0.2** 确认可用的原始数据：`dataset_HW.bin`（40B header + 向量 + bitmap，格式见 [old/test/full_npu/README.md](old/test/full_npu/README.md)）、原始 HW `section.*` / `inverted_union` 目录、`hw_queries.fvecs`、`ground_truth_cache.bin`。
- [ ] **P0.3** 决定索引落盘目录布局（`data_dir` / `data_out`），先用**小子集**（如 10k~100k 文档）跑通，再上全量。
- [ ] **P0.4** 建立端口工作目录 `new/port/`（放 CMake、stubs、harness、数据转换器），`new/` 源码尽量**不改**（只在必要处加 `#ifdef`）。

**验收**：能在服务器上跑通一次 `old/`（`./run.sh -v s`），证明工具链与数据可用。

---

## Phase 1 — 构建骨架（CMake + include 映射 + protobuf）　(本地)(服务器)(低风险)

目标：把 Bazel 依赖图翻译成 CMake，先让**空壳能配置**。

> 已写 [port/CMakeLists.txt](port/CMakeLists.txt)：CANN 探测（复用 old/）+ include-root 符号链接 + ascendc device 库 + protoc 生成 + 三个 host 静态库（common/search/build）+ builder/harness 可执行。**只能在 NPU 服务器 configure**（本地无 CANN）。

- [x] **P1.1** 顶层 `CMakeLists.txt` 已写（位于 `port/`，非 `new/port/`）
- [x] **P1.2** **include-root 映射**已实现并**本地验证通过**：
  - `"src/full_recall/..."` 与 `"full_recall/..."` → CMake 在 build 目录建符号链接指向 `new/`
  - 短前缀 `"utils/..."` `"common/..."` `"configuration/..."` 与长前缀 `"src/utils/..."` 等 → 双 include 根 `port/stubs` + `port/stubs/src` 同时覆盖（已用本地 probe 编译+运行验证）
  - 同目录 `"posting_data.h"` → 编译器自动解析
- [ ] **P1.3** **protobuf 生成**（已在 CMake 中 wire，未在服务器实跑）：对 3 个 `.proto` 跑 `protoc` 生成 `.pb.h/.pb.cc`：
  - [build/proto/document.proto](new/build/proto/document.proto)（`Building.Document/Section/TermInfo/FloatEmbedding`）
  - [build/proto/index_builder.proto](new/build/proto/index_builder.proto)（`IndexBuilderConfig/SectionInfo/SegmentInfo`）
  - [build/proto/index_meta.proto](new/build/proto/index_meta.proto)（`IndexMeta/Config/Schema/Statistics`）

  生成物需可被 `#include "src/full_recall/build/proto/*.pb.h"` 找到（CMake 用 `protoc -I include_root` 生成到 `proto_gen/src/full_recall/build/proto/`）。**确认是否用到 gRPC**（BUILD 里有 `grpc++`）——目前看仅 protobuf 消息，无 service，只需 `libprotobuf`。
- [x] **P1.4** **`securec.h`** 兼容头已写（`memcpy_s`/`memset_s`/`strncpy_s`），路径 `huawei_platform/huawei_secure_c/include/securec.h`
- [x] **P1.5** **`third_party` 映射**：`gflags`/`protobuf` 用 `find_package`；`murmurhash3` 已 vendored 到 `port/stubs/murmurhash/`；`rapidjson` 用 `find_path`（builder 需要）；`jemalloc` 暂不接

**验收**：`cmake` 配置通过；`protoc` 产物生成成功。

---

## Phase 2 — 基建桩（stubs）　(本地)(低风险)

为 ~12 个外部头写**本地最小实现**，放在 include-root 下对应路径。下表是必须提供的符号（已按实际用量核实）：

| 桩文件 | 必须提供的符号 | 说明 |
|---|---|---|
| `src/utils/logger.h` | 宏 `LOG_DEBUG/LOG_INFO/LOG_WARN/LOG_ERROR(msg)`（支持 `<<` 流式）；`Logger::Instance().Init(props)`；`FLAGS_tianjiengine_log_level` | 用量最大（29×）；包一层 `std::cerr` 即可 |
| `src/common/error_code.h` | `ErrorCode::ResultType { SUCCESS, FAIL }` | 14× |
| `src/workflow/async/executor.h` | `class Executor`；`std::shared_ptr<Executor> CreateExecutor()`；`Executor::CreateExecuteContext(LogContext&)` 返回 context；`ctx->AddTask(std::function<ErrorCode::ResultType()>)`；`ctx->Wait(std::function<void(ErrorCode::ResultType)>)`；`class LogContext` | 并发原语。**先做同步版**（AddTask 立即执行并存结果，Wait 回放）跑通，再换线程池 |
| `src/utils/performance_recorder.h` | `RecordGuard{std::string}`（RAII 计时） | 22×；可先做空实现，后填计时 |
| `src/utils/file_manager.h` | 文件读写辅助（builder/data 落盘与 mmap 加载用）| 8×；**需读 build/file 与 data/*.cpp 确认具体 API** |
| `src/configuration/develop_configuration.h` + `configuration/develop_configuration.h` | 下列 `FLAGS_*` 的 `DECLARE/DEFINE` | 见下方 FLAGS 清单；用 gflags 定义默认值 |
| `src/common/uncopyable.h` | `TIANJIENGINE_DECLARE_UNCOPYABLE(Class)` 宏 | 删拷贝构造/赋值 |
| `src/common/tianjiengine_constants.h` | 工程常量 | **需 grep 确认用到哪些** |
| `src/utils/string_util.h` | 字符串工具（2×） | 按报错补 |
| `src/utils/random_number.h` / `alarm.h` / `time_util.h` / `threading/thread_pool.h` / `single_thread_task.h` | 各 1× | 多半仅 builder/次要路径用；按报错**惰性补** |

> 落地位置：独立顶层目录 **`port/stubs/`**（与 old/、new/ 平级，new/ 不改）。已在本地用 Apple clang 对 new/ 纯 host 切片（trans_number/hash/codec + murmurhash）编译验证通过。

- [x] **P2.1** `logger.h`（含 log4cplus level shim + `Logger::GetLogLevel`）— 本地验证通过
- [x] **P2.2** `error_code.h` + `uncopyable.h` + `tianjiengine_constants.h`（`INDEX_SHARD_PREFIX_NUMS`/`NPU_MEMORY_WARNING_THRESHOLD`）
- [x] **P2.3** `executor.h`（同步版）+ `LogContext`
- [x] **P2.4** `performance_recorder.h`（`RecordGuard`，`TIANJI_PERF=1` 打印耗时）
- [x] **P2.5** `develop_configuration.h`(DECLARE) + `.cpp`(DEFINE) 全部 `FLAGS_*`（gflags）；含 `configuration/` 无前缀转发头
- [x] **P2.6** `file_manager.h`（`GetRealFilePath`）+ `string_util.h`（`StringSplit`/`StringToNumber`）
- [x] **P2.附** `securec.h`（`memcpy_s`/`memset_s`/`strncpy_s`，真实 include 路径 `huawei_platform/huawei_secure_c/include/`）+ `murmurhash/MurmurHash3`（公有领域）+ `alarm.h`
- [x] **P2.7a** 编写 CMake 时发现并补齐的短前缀桩：`utils/safe_unordered_map.h`（`Find`/`Insert`，gm_memory_manager 用）、`utils/rapidjson_util.h`（`GetStringFromValueObj`/`GetUintFromValueObj`，builder 用）
- [ ] **P2.7b** 其余 utils 头按 Phase 4 编译报错惰性补：`random_number.h` `time_util.h` `threading/thread_pool.h` `single_thread_task.h`（多为 process/ 或次要路径）

> 全部桩已用本地 probe（`port` 外的临时 TU）编译+运行验证：logger/executor/error_code/uncopyable/performance_recorder/file_manager/string_util/safe_unordered_map/alarm/securec/murmurhash 均 OK；`develop_configuration.cpp` 依赖 gflags 本地未验（服务器编）。

**必须定义的 `FLAGS_*`**（gflags）：
```
full_recall_batch_search_thread_num, full_recall_batch_accumulation_max_size,
full_recall_batch_accumulation_time_interval, full_recall_aggregator_block_dim,
full_recall_scorer_block_dim, full_recall_text_filter_block_dim,
full_recall_bitlist_denseness_threshold, full_recall_device_load_rate,
full_recall_npu_topk_enters_threshold_ratio, full_recall_npu_topk_finish_buffer_ratio,
full_recall_npu_topk_loop_count, full_recall_search_limit,
full_recall_send_batcher_only_score, full_recall_stream_init_size,
build_from_memory, data_dir, data_out, ipc_socket_path,
server_log_properties, tianjiengine_log_level
```
> 注意：这些 `block_dim` / `ratio` / `threshold` 的**默认值影响正确性与性能**，需从华为文档或 kernel 代码推断合理值（Phase 7 调参）。

**验收**：所有桩头编译通过（空 `main` 引用它们能过）。

---

## Phase 3 — 设备侧 kernel 编译　(服务器)(低风险)　✅ 已完成

- [x] **P3.1** 顶层 CMake 内联 `ascendc_library(ascend_device ...)`，服务器上成功编 `libascend_device.so`
- [x] **P3.2** 4 个 kernel 全部编过并打包（aic/aiv/host 三路 + `ascendc_pack_kernel`）
- [x] **P3.3** device.so 接入顶层 CMake，searcher 依赖它 + 其生成的 `aclrtlaunch_*.h`（auto_gen 目录）

**验收**：✅ `libascend_device.so` 在 NPU 服务器生成成功。

---

## Phase 4 — Host 库编译连通（探针）　(服务器)　✅ 已通过

> **2026-07-03 全链路编译+链接成功**：`build/fr_search`（631KB）、`build/fr_converter`、`libascend_device.so` 均产出。整条 `new/` 搜索链路可执行。
> 收敛记录（依赖真实环境暴露的问题）：
> - `new/build/` 被 `.gitignore` 的 `**/build/` 误伤 → 加负例重新纳入
> - GCC12 严格 → 系统性缺 `<algorithm>/<limits>/<numeric>` → 用 `-include frnew_prelude.h` 强制包含
> - `data_table_repository.cpp`（仅 process/ 用，缺 `Parse*` 符号）→ 排除
> - 设备 kernel launch 头在 `build/include/ascend_device/`（非 auto_gen）
> - `LOG_TRACE`/`random_number.h` 补桩；`doc_id_mapping.h` 补 `<limits>`
> - CANN 无 `libtiling.so`（只有 `tiling_api`）→ 链接列表去掉 `tiling`，加 `-L<CANN>/lib64` + rpath
> - protobuf 4.25 依赖 absl，且本机 protobuf CONFIG 包损坏 → 强制 MODULE + `-Wl,--copy-dt-needed-entries`（经 libprotobuf.so 的 DT_NEEDED 解析 absl）


> 这是**信息量最大的一步**：先不求跑对，只求 `build/data/codec/common/search` 能编过、能链接，快速暴露「抽取到底缺多少」。

按依赖自底向上编（顺序即 Bazel 依赖图）：

- [x] **P4.1** `common/` + `common/number/trans_number` + `common/hash`（murmurhash3）——**编过并链成 `libfrnew_common.a`** ✅
- [x] **P4.2** `codec/`（`posting_encoder`、`head_decoder`）——含于 frnew_common ✅
- [ ] **P4.3** `build/file/` + `build/schema` + `build/proto`（builder，需 rapidjson-devel；proto 已生成 ✅）
- [~] **P4.4** `data/`——编译中；已修 `doc_id_mapping.h` 缺 `<limits>`、`data_table_repository` 缺 `random_number.h` 桩
- [~] **P4.5** `search/query/`——已修 `LOG_TRACE` 缺失
- [~] **P4.6** `search/searcher/`——已修 `ascend_device/aclrtlaunch_*.h` include 路径（auto_gen）；链 `ascendcl/register/tiling/...`

> 已通过的：proto 生成、`frnew_common` 编+链、`fr_converter` 编+链、`libascend_device.so` 编+打包。
> 第 2 轮修复（4 项）：`LOG_TRACE` 桩、`random_number.h` 桩、`doc_id_mapping.h` 加 `<limits>`、frnew_search include auto_gen。

> `data_table_repository` 用 `ASCEND_C_ENABLED` 走真实 `DataTable` 分支（已在 CMake 定义）。

**验收**：host 库全部编译 + 链接通过。**缺失符号在此暴露** → 回填桩或标记「抽取缺文件」。

---

## Phase 5 — 数据转换器（dataset → builder 输入）　(本地)(服务器)(中风险)

builder 输入 = `schema.json`（JSON of `IndexBuilderConfig`）+ 每字段 length-delimited 输入流。已读源码确认契约并写成转换器 [port/converter/build_input.cpp](port/converter/build_input.cpp)（本地 fake-proto syntax-check 通过）。

**已确认的输入契约**：
- schema.json = `Building.IndexBuilderConfig` 的 JSON（`JsonStringToMessage` 解析，camelCase）；`main` 读 `<data_dir>/schema.json`
- 每字段输入文件（`ReadAndDoTask` 格式）：`[u32 count]` + 每 doc `[u32 len][u64 gdocid][len-8 字节 pb]`
- 目录布局：`<dir>/docid/attachment.docid.<seg>`（gdocid，pb 空）、`<dir>/content/section.content.<seg>`（pb=`Building.Section`）
- 一个 section 同时 `docIndex+vecIndex`：`Section.float_embedding[0]`=64×FP32 向量，`Section.terminfos[*].uint64Value`=原始 tag id
- **token 一致性**：builder 落盘 tokenId = `Hash64(std::to_string(tagId))`；查询侧 `TermNode` token 必须同法（harness 已按此实现）
- 向量落盘：FP32→FP16，按 `DIM_PER_BLOCK=16` 维块 + `split_doc_num_zn` 文档块的 Zn 布局；`doc_num_per_segment` 必须是 16 的倍数

- [x] **P5.1/5.2** 读 `builder_impl`/`*_builder`/`builder_schema`/`file_reader`/`inverted_builder`，输入契约与落盘格式已确认（见上）
- [x] **P5.3** 转换器已写：支持 `--mode synthetic`（随机数据，无需 47GB 集即可打通链路）与 `--mode hw`（读 `dataset_HW.bin` 子集）；输出 schema.json + 分段文件
- [ ] **P5.4** 服务器跑 builder 产出索引（先 `--mode synthetic --docs 10000`，再 hw 子集）
- [ ] **P5.5** 交叉校验 doc 数/segment 数/向量/tag 集合

**验收**：builder 在子集上成功产出全套索引文件，且抽样校验一致。

---

## Phase 6 — 驱动 harness（绕过 process/）　(本地)(服务器)(中风险)

独立 `main` [port/harness/main.cpp](port/harness/main.cpp)（**不用 SearchBatcher/FullSearchProxy/IndexManager**），已写：

- [x] **P6.1** `aclInit` → `DataTable::LoadData(deviceId, index_dir)`（上传显存），打印 docNum/segments/加载耗时
- [x] **P6.2** 查询：读 `.fvecs` 向量；构造 `QueryNode` 树——**空 `AndNode`=match-all**（text filter memset 0xff），`--filter_tag T>=0` 时挂一个 `TermNode(content, Hash64("T"))` 走过滤路径
- [x] **P6.3** 逐 batch 新建 `FullRecallSearcher{deviceId, dataTable}`：`AddQuery(tree, vec, topK)` → `BatchSearch(results, vec_field, isMultiShard=false)`（每 batch 新建，因 searcher 累积成员不可复用）
- [x] **P6.4** 读 `FullRecallResult.{docIds,scores}`（单分片模式聚合器已回填），打印首查询 Top-10 + 延迟 avg/p50/p99
- [x] **P6.5** 接入顶层 CMake，产出可执行 `fr_search`

**验收**：单查询端到端不崩、返回 topK 结果。（服务器实机验证，需 Phase 4 先编过。）

> **2026-07-03 端到端跑通 ✅**：合成数据（4096 docs / 4 seg / dim64）上 `fr_converter → fr_builder(exit=0) → fr_search` 全通。日志确认 NPU 上依次执行：MMad 打分（`finish vector scoring by mmad`）、bitmap 过滤（match-all）、NPU TopK（`NPU TOPK effectiveCount=300`）、聚合（`Aggregator and topK success`），返回结果 + 延迟统计。
> 踩到并修正的运行期约束：`split_doc_num_zn <= 16384/dim`（dim64→256），已改 converter 默认值。
> 待观察（非阻塞）：日志多次 `[WARN] stream is NULL`（StreamManager 返回空流，kernel 用默认流仍执行）——Phase 7/8 再查。
> 合成数据的召回/延迟无参考意义；真实召回(≥99%)与 P99(<2ms) 属 Phase 7/8，需真实数据集 + ground truth。

---

## Phase 7 — 端到端跑通 + 召回验证　(服务器)(需实机)

> 采用 old/ 的同一套数据（dataset_HW.bin + hw_queries.fvecs + filter_expr_600.txt，均为 old/ gen_vec/gen_bitmap 合成）。harness 已扩展：headered fvecs 加载、filter 表达式解析→QueryNode 树（NOT>AND>OR，本地测过）、filter 随机分配、**CPU 暴力自校验召回**（mmap dataset_HW.bin，FP16 对齐的内积 top-K，绕过 old/ GT 格式与随机种子）。converter `--mode hw` 改 mmap 避免 45G 全读。

- [x] **P7.0 代码审查**：核对 new/ 打分/聚合/过滤语义与 CPU 参考一致——MMad=内积、`compareDesc`=降序、docId=gdocid、tag/token/向量编码全链一致。**发现一个关键点**：new/ 对「token 在索引中不存在」的 term 会**丢弃**（AND 丢弃=恒真；OR 全缺组→`[OR,0,0]`→CheckPostExpr 失败→查询报错）。CPU 参考把缺失 tag 当 false → **仅当索引含全部 filter tag（即全量数据）时两者一致**。子集 + filter 会出现伪不一致甚至查询报错。
- [~] **P7.1** harness 支持真实 query+filter+CPU 召回自校验（bitmap 位测过滤，可全量暴力）
- [ ] **P7.2a 打分正确性**（子集即可）：**match-all（不加 --filter_file）** 跑召回自校验，隔离验证 MMad 打分+TopK+聚合，期望 ≈100%
- [ ] **P7.2b 过滤正确性**（**必须全量索引** `--docs 0`）：加 --filter_file 跑召回，全量下所有 filter tag 都在 → new/ 与 CPU 一致，期望 ≈100%
- [ ] **P7.3** 若召回偏低：查 FP16 精度/topk 阈值边界（NPU TopK 在候选>阈值时预筛，`npu_topk_*` 参数）
- [ ] **P7.4** 全量真实 P99 延迟（关掉 recall），对比 old/ 的 6–8ms

**验收**：全量数据上平均召回率 ≥ 99%。

---

## Phase 8 — 性能测量（验证「思路是否更快」）　(服务器)(需实机)

- [ ] **P8.1** 打点 score / filter / aggregator / topk 各阶段延迟（RecordGuard）。
- [ ] **P8.2** 统计 **P99 查询延迟**，与目标 2ms、与 `old/` 的 6–8ms 对比。
- [ ] **P8.3** 出结论：full-recall 思路能否达标 → 决定是否继续投入（补 `process/` 攒批、多 shard、多 device 扩展）。

**最终验收**：给出「new/ 能跑 + 召回≥99% + P99 延迟数字」的结论报告。

---

## 附录 A — 外部依赖清单（Bazel target → 处理方式）

| Bazel target | 用途 | 路线 A 处理 |
|---|---|---|
| `@ascend//:acl` `@ascend//:ascend` | CANN/ACL + AscendC | 复用 old/ 的 CANN 探测，链 `ascendcl` 等（服务器） |
| `//src/full_recall/search/device:ascend_device` | 4 个 kernel | 用现成 device/CMakeLists.txt 编 .so |
| `//src/utils:tianji_log` / `tianjiengine_utils` | 日志/工具 | **桩**（Phase 2） |
| `//src/utils:file_manager` | 文件 IO | **桩**（确认 API） |
| `//src/workflow/async:executor` | 异步执行 | **桩**（同步版起步） |
| `//src/configuration:*` | gflags 配置 | **桩**（定义 FLAGS） |
| `//src/common:tianjiengine_common` | error_code/uncopyable/常量 | **桩** |
| `//third_party:grpc++` | protobuf(+grpc?) | protoc 生成；确认无 service |
| `//third_party:gflags` | 命令行 | 系统包 |
| `//third_party:murmurhash3` | hash | 取上游单文件 |
| `//third_party:jemalloc` | 分配器 | 可选，先跳过 |
| `//huawei_platform:huaweiSecureC` | `memcpy_s` 等 | 兼容 `securec.h` 或 libsecurec |
| `//src/vector_recall/*` `//src/full_recall/process:*` | 在线服务/攒批/IndexManager | **绕过**（不编 process/） |

## 附录 B — 目录结构建议

```
<repo>/
  new/                      # 华为源码，原样不改（build/ codec/ common/ data/ process/(暂不编) search/）
  port/                     # 本 TODO 的新增工作，独立顶层目录，与 old/ new/ 平级
    CMakeLists.txt          # 顶层：CANN 探测 + include 映射 + 汇总各库（待写）
    stubs/                  # Phase 2 基建桩（已完成核心集）
      src/utils/*  src/common/*  src/configuration/*  src/workflow/async/*
      configuration/develop_configuration.h        # 无 src 前缀转发头
      murmurhash/MurmurHash3.{h,cpp}
      huawei_platform/huawei_secure_c/include/securec.h
    proto_gen/              # protoc 产物 .pb.h/.pb.cc（待生成）
    converter/              # Phase 5 数据转换器
    harness/main.cpp        # Phase 6 驱动
    # include-root 符号链接（src/full_recall->new, full_recall->new）由 CMake 在 build 目录生成
```

## 附录 C — 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| `new/` 抽取不完整，Phase 4 暴露缺失 .cpp/私有依赖 | 需实机 | Phase 4 探针尽早做；缺失项及时向华为方索要或桩化 |
| 数据转换与 builder 编码约定不符（Zn 对齐/posting 编码/segment 切分）| 中 | Phase 5.5 抽样对拍；细读 codec/ + *_field_data |
| `Executor` 异步语义（stream/显存生命周期）同步版跑不对 | 中 | 先同步跑通正确性，再换线程池对齐并发行为 |
| `FLAGS_*` 默认值（block_dim/topk 阈值）影响正确性/性能 | 中 | Phase 7 系统调参；从 kernel tiling 反推 |
| 本地无 NPU，编译/运行只能服务器迭代，回合慢 | 需实机 | 本地做全部可做项（CMake/桩/转换器/harness 代码/protoc）；服务器只做编译链接与运行 |
| `gm_memory_manager`/`stream_manager` 与实机 CANN 版本契合度 | 需实机 | 实机验证，必要时按 CANN 版本微调 |

## 建议执行顺序（并行化）

```
本地 可先做：Phase 0.4, 1, 2, 5(转换器代码), 6(harness代码)
服务器 并行：Phase 3(device.so，最独立)
汇合：Phase 4(编译连通探针) → 5(产索引) → 6(联调) → 7(召回) → 8(性能)
```
关键路径的第一个里程碑 = **Phase 4 编译连通**，它决定「抽取代码到底缺多少」，应优先冲刺。
