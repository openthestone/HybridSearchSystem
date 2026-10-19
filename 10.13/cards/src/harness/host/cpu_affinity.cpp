// Before any libc header: glibc gates pthread_setaffinity_np / CPU_* on it.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "cpu_affinity.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace npur_harness {

// The cards do not share a NUMA node (card 0 off node4, card 1 off node2 on the 2-card box), so
// a process-wide `taskset` is remote to at least one of them and its H2D/D2H crosses QPI.
#ifdef __linux__
static std::vector<cpu_set_t> g_cardCpus;

static bool ParseCpuList(const std::string& s, cpu_set_t& out) {
    CPU_ZERO(&out);
    bool any = false;
    size_t i = 0;
    while (i < s.size()) {
        size_t comma = s.find(',', i);
        std::string tok = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        i = (comma == std::string::npos) ? s.size() : comma + 1;
        if (tok.empty())
            continue;
        const size_t dash = tok.find('-');
        char* endp = nullptr;
        const long lo = std::strtol(tok.c_str(), &endp, 10);
        long hi = lo;
        if (dash != std::string::npos)
            hi = std::strtol(tok.c_str() + dash + 1, &endp, 10);
        if (lo < 0 || hi < lo || hi >= CPU_SETSIZE)
            return false;
        for (long c = lo; c <= hi; ++c) {
            CPU_SET(static_cast<int>(c), &out);
            any = true;
        }
    }
    return any;
}

void PinThreadToCard(size_t cardIdx) {
    if (cardIdx >= g_cardCpus.size())
        return;
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_cardCpus[cardIdx]);
}

void PinThreadToCards(size_t firstCard, size_t count) {
    if (g_cardCpus.empty() || firstCard + count > g_cardCpus.size())
        return;
    cpu_set_t u;
    CPU_ZERO(&u);
    for (size_t c = firstCard; c < firstCard + count; ++c)
        CPU_OR(&u, &u, &g_cardCpus[c]);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &u);
}
#else
void PinThreadToCard(size_t) {}
void PinThreadToCards(size_t, size_t) {}
#endif

std::vector<std::string> SplitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t b = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (b != std::string::npos)
            out.push_back(tok.substr(b, e - b + 1));
    }
    return out;
}

#ifdef __linux__
bool SetCardCpus(const std::string& spec, const std::vector<int32_t>& deviceIds, std::string* error) {
    if (spec.empty())
        return true;
    std::vector<std::string> lists;
    for (size_t i = 0, j; i <= spec.size(); i = j + 1) {
        j = spec.find(';', i);
        if (j == std::string::npos)
            j = spec.size();
        lists.push_back(spec.substr(i, j - i));
    }
    if (lists.size() != deviceIds.size()) {
        if (error)
            *error = "--card_cpus has " + std::to_string(lists.size()) + " core list(s) but there are " +
                     std::to_string(deviceIds.size()) + " card(s); give one ';'-separated list per --device_ids entry";
        return false;
    }
    g_cardCpus.resize(lists.size());
    for (size_t c = 0; c < lists.size(); ++c) {
        if (!ParseCpuList(lists[c], g_cardCpus[c])) {
            if (error)
                *error = "--card_cpus entry " + std::to_string(c) + " (\"" + lists[c] + "\") is not a valid core list";
            g_cardCpus.clear();
            return false;
        }
    }
    if (g_cardCpus.size() == 1)
        PinThreadToCard(0);
    std::printf("[AFFINITY] per-card pinning:");
    for (size_t c = 0; c < lists.size(); ++c)
        std::printf(" card%zu(dev%d)=%s", c, deviceIds[c], lists[c].c_str());
    std::printf("\n");
    return true;
}
#else
bool SetCardCpus(const std::string&, const std::vector<int32_t>&, std::string*) {
    return true;  // --card_cpus is silently ignored off Linux, as it always was
}
#endif

}  // namespace npur_harness
