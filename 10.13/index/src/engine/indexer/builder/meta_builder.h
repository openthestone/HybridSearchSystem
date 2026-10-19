#pragma once
#include <ostream>
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/indexer/proto/index_meta.pb.h"

namespace NpuRetrieval {

class MetaBuilder {
   public:
    bool Build(BuilderSchema& schema, const std::string& realOutputDir);

   private:
    bool WriteMeta(const std::string& outputDir, Building::Meta::IndexMeta& indexMeta);
    uint8_t m_version = 0;
};
}  // namespace NpuRetrieval
