#pragma once

namespace NpuRetrieval {
// On-disk posting layout tag. The numeric values are written into the 4-bit type field of every
// posting header, so they are a format contract and must not be reordered or renumbered.
// SPARSE_PACKED is SPARSE_BITMAP with the two payload sections folded together: one uint32 per
// non-zero unit holding [unitIndex:16][mask:16]. UNKNOWN is a sentinel, never written to disk.
enum class PostingLayout { DENSE_BITMAP = 0, SPARSE_BITMAP, DENSE_BYTEMAP, SPARSE_IDLIST, SPARSE_PACKED, UNKNOWN };
}  // namespace NpuRetrieval
