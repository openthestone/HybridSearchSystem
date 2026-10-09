#pragma once

#include <cstdint>
#include <vector>
#include <fstream>
#include <memory>
#include "acl/acl.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"

namespace NpuRetrieval {
class VectorFieldData {
   public:
    VectorFieldData() {}

    ~VectorFieldData() {
        if (m_deviceData != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(m_deviceData));
        }
    }

    // Called once per segment, in order. size: the file size.
    bool AddSegment(std::ifstream& inputStream, uint64_t size);

    // Copy the accumulated data to the device.
    bool FinishAdd();

    uint32_t GetDimension() const {
        return m_dimension;
    }

    uint64_t GetLength() const {
        return m_totalLength;
    }

    uint8_t* GetFieldDataInDevice() const {
        return m_deviceData;
    }

   private:
    bool UploadToDevice();

    std::vector<char> m_hostBytes{};  // all segments concatenated, host side
    uint8_t* m_deviceData{};
    uint64_t m_totalLength{};
    uint32_t m_dimension{};
};
}  // namespace NpuRetrieval
