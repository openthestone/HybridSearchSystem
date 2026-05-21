#pragma once

#include <cstdint>
#include <vector>
#include "../BatchTask.h"
#include "kernel_operator.h"

// 控制日志级别的打印开关
#define DEBUG_LOG_ENABLE false
#define ERROR_LOG_ENABLE false
#define DUMP_TENSOR_ENABLE false

// 这里多一层宏定义的嵌套是为了，彻底关闭日志打印。
// AscendC::printf是设计是，只要代码中存在AscendC::printf，即使判断条件使得AscendC::printf永远无法执行，
// 其内核在运行到其AscendC::printf所在函数时还是会打印出内核信息：如下所示
// opType=PostingBitListToSet, DumpHead: AIC-0, CoreType=MIX, block dim=8, total_block_num=8, block_remain_len=1048456,
// block_initial_space=1048576, rsv=0, magic=5aa5bccd
// CANN Version: 8.0.0, TimeStamp: 20241231000821303
// 因此这里在日志关闭时，使用宏定义直接把AscendC::printf的调用代码去除
#if DEBUG_LOG_ENABLE == true
#define DEBUG_LOG(fmt, args...) AscendC::printf(fmt, ##args)
#else
#define DEBUG_LOG(fmt, args...)
#endif

#if DUMP_TENSOR_ENABLE == true
#define DEBUG_DUMP_TENSOR(tensor, desc, dumpSize) AscendC::DumpTensor(tensor, desc, dumpSize)
#else
#define DEBUG_DUMP_TENSOR(tensor, desc, dumpSize)
#endif

#if ERROR_LOG_ENABLE == true
#define ERROR_LOG(fmt, args...) AscendC::printf(fmt, ##args)
#else
#define ERROR_LOG(fmt, args...)
#endif