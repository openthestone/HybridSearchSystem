#include "utils/ProtoParser.h"

#include <limits>
#include <string>

#include "doc.pb.h"

namespace ProtoParser {

bool ParseFirstFloatEmbedding(const uint8_t *data, size_t size, std::vector<float> &embedding)
{
    embedding.clear();
    if (data == nullptr || size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    Falcon::IndexFactory::Section section;
    if (!section.ParseFromArray(data, static_cast<int>(size)) ||
        section.float_embedding_size() <= 0)
    {
        return false;
    }

    const Falcon::IndexFactory::FloatEmbedding &first = section.float_embedding(0);
    embedding.reserve(static_cast<size_t>(first.embedding_size()));
    for (float value : first.embedding())
    {
        embedding.push_back(value);
    }
    return !embedding.empty();
}

bool ForEachTerm(const uint8_t *data, size_t size, TermCallback callback, void *user_data)
{
    if (data == nullptr ||
        callback == nullptr ||
        size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    Falcon::IndexFactory::Section section;
    if (!section.ParseFromArray(data, static_cast<int>(size)))
    {
        return false;
    }

    for (const auto &term_info : section.terminfos())
    {
        const std::string &term = term_info.term();
        if (!term.empty())
        {
            callback(term.data(), term.size(), user_data);
        }
    }
    return true;
}

} // namespace ProtoParser
