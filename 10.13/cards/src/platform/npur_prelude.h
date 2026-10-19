/*
 * npur_prelude.h — force-included (via `-include`) into every translation unit
 * that compiles extracted `engine/` sources.
 *
 * The NpuRetrieval full-recall code was sliced out of a larger tree and relies
 * on a number of standard headers being pulled in transitively (they were, in
 * the original build, via other project headers we don't compile). GCC 12 with
 * a stricter libstdc++ then fails with "X is not a member of std" for things
 * like std::sort / std::numeric_limits / std::accumulate.
 *
 * Rather than editing many Huawei source files, we make these ubiquitous
 * standard headers available everywhere. This does not change behaviour — it
 * only guarantees the declarations the code already assumes are visible.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
