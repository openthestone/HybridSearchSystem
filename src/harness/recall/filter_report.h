/*
 * filter_report.h -- the --effective_filter_file diagnostic: for the first few queries, what the
 * engine actually ends up evaluating. A term whose token is absent from the index is DROPPED by
 * the engine, so the filter that runs is not always the filter that was written.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../query/filter_setup.h"
#include "src/full_recall/index/data_table.h"

namespace npur_harness {

// tag_doc_freq.txt as emitted by fr_converter; used only to annotate the report.
bool LoadConvertedTagFreqFile(const std::string& path, std::vector<uint64_t>& tagDocFreq);

// No-op when `path` is empty.
bool WriteEffectiveFilterExpressions(const std::string& path, const std::shared_ptr<NpuRetrieval::DataTable>& dataTable,
                                     const FilterSet& filters, bool haveFilters, const std::vector<uint32_t>& topks,
                                     size_t nq, const std::string& postingField,
                                     const std::vector<uint64_t>* convertedTagDocFreq,
                                     const std::string& convertedTagFreqPath, const std::string& indexDir);

// One call for the whole --effective_filter_file step: reads the optional
// --converted_tag_freq_file annotation, then writes the report. A no-op when the flag is unset.
bool ReportEffectiveFilters(const std::shared_ptr<NpuRetrieval::DataTable>& dataTable, const FilterSet& filters,
                            bool haveFilters, const std::vector<uint32_t>& topks, size_t nq,
                            const std::string& indexDir);

}  // namespace npur_harness
