#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "acl/acl.h"
#include "src/utils/logger.h"

namespace NpuRetrieval {
#define CHECK_ACL_ONLY_LOG(x)                                                              \
    do {                                                                                   \
        aclError code = x;                                                                 \
        if (code != ACL_SUCCESS) {                                                         \
            LOG_ERROR("NPU aclrt Error " << code << ", " << __FILE__ << ", " << __LINE__); \
        }                                                                                  \
    } while (0)
}  // namespace NpuRetrieval
