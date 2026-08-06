#pragma once

namespace NpuRetrieval {
// On-disk posting layout tag. The numeric values are written into the 4-bit
// type field of every posting header, so they are a format contract and must
// not be reordered or renumbered.
enum class PostingLayout { DENSE_BITMAP = 0, SPARSE_BITMAP, DENSE_BYTEMAP, SPARSE_IDLIST, UNKNOWN };
}  // namespace NpuRetrieval
