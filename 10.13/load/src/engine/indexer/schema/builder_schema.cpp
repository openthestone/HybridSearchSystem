#include "builder_schema.h"
#include "google/protobuf/json/json.h"
#include "src/utils/logger.h"

namespace NpuRetrieval {
bool BuilderSchema::ParseAttachment(const Building::IndexBuilderConfig& config) {
    bool hasDocId = false;
    for (auto& attachment : config.attachment_infos()) {
        if (attachment.name().empty()) {
            LOG_ERROR("invalid attachment, name is empty, discarded");
            continue;
        }
        // docid file
        if (attachment.is_docid()) {
            hasDocId = true;
            m_docIdInfo.name = attachment.name();
            m_docIdInfo.isSdocid = attachment.is_sdocid();
            break;
        }
        // add handling for other forward-index files here when needed
    }
    if (!hasDocId) {
        LOG_ERROR("do not have docid file, please check");
        return false;
    }
    return true;
}

bool BuilderSchema::ParseSection(const Building::IndexBuilderConfig& config) {
    for (auto& section : config.section_infos()) {
        if (section.name().empty()) {
            LOG_ERROR("invalid section, name is empty, discarded");
            continue;
        }
        // posting (inverted) field
        if (section.doc_index()) {
            m_inverted[section.name()] = section.inverted_info();
        }
        // vector field
        if (section.vec_index()) {
            if (section.vector_info().dimension() == 0) {
                LOG_ERROR("invalid vector dimension 0, name is:" << section.name());
                return false;
            }
            if (section.vector_info().dimension() > UINT16_MAX) {
                LOG_ERROR("invalid vector dimension larger than UINT16_MAX, dimension is:"
                          << section.vector_info().dimension());
                return false;
            }
            m_embedding[section.name()] = static_cast<uint16_t>(section.vector_info().dimension());
        }
    }
    return true;
}

bool BuilderSchema::Initialize(const std::string& data) {
    if (data.empty()) {
        return false;
    }
    google::protobuf::json::ParseOptions deserializeOptions;
    Building::IndexBuilderConfig config;
    auto parseStatus = google::protobuf::json::JsonStringToMessage(data, &config, deserializeOptions);
    if (!parseStatus.ok()) {
        LOG_ERROR("fail to parse, errorCode=" << parseStatus.code() << ", errorMsg=" << parseStatus.message()
                                              << ", data " << data);
        return false;
    }
    LOG_INFO("parse shcema result: " << config.ShortDebugString());
    if (config.version() > UINT8_MAX) {
        LOG_ERROR("version need to be uint8, please check");
        return false;
    }
    if (!ParseAttachment(config)) {
        LOG_ERROR("Parseattachment fail, please check");
        return false;
    }
    if (!ParseSection(config)) {
        LOG_ERROR("ParseSection fail, please check");
        return false;
    }
    m_version = static_cast<uint8_t>(config.version());
    m_segmentNum = config.segment_info().segment_num();
    m_docNumPerSegment = config.segment_info().doc_num_per_segment();
    m_docNum = config.segment_info().doc_num();
    m_splitDocNumZn = config.segment_info().split_doc_num_zn();
    return true;
}
}  // namespace NpuRetrieval
