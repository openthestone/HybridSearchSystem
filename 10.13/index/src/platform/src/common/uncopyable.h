/*
 * Stub for NpuRetrieval `src/common/uncopyable.h`.
 *
 * Used as `NPURETRIEVAL_DECLARE_UNCOPYABLE(ClassName);` inside class bodies
 * (e.g. DataTable) to delete copy construction / assignment.
 */
#pragma once

#define NPURETRIEVAL_DECLARE_UNCOPYABLE(ClassName) \
    ClassName(const ClassName&) = delete;          \
    ClassName& operator=(const ClassName&) = delete
