// fr_loadbench: 单独量"把 N 字节切成每卡一份、并行搬到 NPU 上"要多久。
//
// 验收里"索引加载时间"这一项问的就是这件事，但 fr_search 的加载还夹着建表、解字典、
// 组织分片等一堆引擎自己的工作。这个程序不碰索引、不碰引擎，只做两件事：
//   读盘   把源文件读进主机内存（对应 fr_search 日志里的 bulk）
//   拷到设备  aclrtMemcpy（对应日志里的 H2D）
// 每张卡一个线程同时开跑，总耗时取最慢的那张。
//
//   fr_loadbench --devices 0,1 --gib_per_card 16          数据来自内存，只测拷到设备这一步
//   fr_loadbench --devices 0,1 --files /path/a,/path/b    读这两个文件，尺寸按文件实际大小
//   fr_loadbench --devices 0,1 --dirs .../shard0,.../shard1 --gib_per_card 16
//                                                         读索引分片目录，每卡读满 16 GiB 就停
//   fr_loadbench --devices 0,1 --gib_per_card 16 --repeat 3
//
// 主机内存要够：每卡一份完整缓冲，两卡 16GiB 就是 32GiB 常驻。
#include <acl/acl.h>
#include <dirent.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <fcntl.h>
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(devices, "0,1", "逗号分隔的 NPU 设备号，每个设备一份数据一个线程");
DEFINE_double(gib_per_card, 16.0, "每卡搬多少 GiB；给了 --files 时以文件实际大小为准");
DEFINE_string(files, "", "逗号分隔的源文件，每卡一个。留空则数据来自内存，只测拷到设备这一步");
DEFINE_string(dirs, "", "逗号分隔的索引分片目录，每卡一个；按文件名顺序读满 --gib_per_card 就停");
DEFINE_int32(repeat, 3, "重复几轮，每轮重新分配、重新读");
DEFINE_int32(read_chunk_mib, 64, "读盘的单次 pread 大小（MiB）");
DEFINE_string(card_cpus, "auto",
              "每卡一份核列表，';' 分隔、和 --devices 一一对应，如 0-23;24-47。"
              "auto = 按 NUMA 节点依次分配；空 = 不绑核。两张卡挤同一个节点会明显变慢");
DEFINE_int32(read_threads, 4,
             "每张卡用几个线程读盘。单线程 copy_to_user 约 0.9 GB/s 就到顶，分文件并行约 4 倍");

namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
double GiBps(size_t bytes, double ms) {
    return ms > 0 ? (static_cast<double>(bytes) / 1073741824.0) / (ms / 1000.0) : 0.0;
}

std::vector<std::string> Split(const std::string& s) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= s.size()) {
        const size_t comma = s.find(',', begin);
        const std::string piece = s.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!piece.empty())
            out.push_back(piece);
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    return out;
}

// 从一个分片目录里按文件名顺序取够 quota 字节：每项是 (路径, 从这个文件里读多少)。
// 最后一个文件通常只读一部分，到量就停。
struct Piece {
    std::string path;
    size_t bytes = 0;
    size_t dstOffset = 0;   // 在主机缓冲里的落点，这样多个线程可以同时往不同位置写
};

void ListFiles(const std::string& dir, std::vector<std::string>* out) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr)
        return;
    std::vector<std::string> subdirs;
    while (const dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..")
            continue;
        const std::string full = dir + "/" + name;
        struct stat st {};
        if (::stat(full.c_str(), &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode))
            subdirs.push_back(full);
        else if (S_ISREG(st.st_mode) && st.st_size > 0)
            out->push_back(full);
    }
    ::closedir(d);
    for (const std::string& sub : subdirs)
        ListFiles(sub, out);
}

std::vector<Piece> TakeFrom(const std::string& dir, size_t quota, size_t* got) {
    std::vector<std::string> files;
    ListFiles(dir, &files);
    std::sort(files.begin(), files.end());
    std::vector<Piece> pieces;
    *got = 0;
    for (const std::string& f : files) {
        if (*got >= quota)
            break;
        struct stat st {};
        if (::stat(f.c_str(), &st) != 0)
            continue;
        const size_t take = std::min(static_cast<size_t>(st.st_size), quota - *got);
        pieces.push_back({f, take, *got});
        *got += take;
    }
    return pieces;
}

// 绑核：止住线程迁移。实测两张卡分到不同 NUMA 节点时，双卡并行等于各自单跑；
// 挤在同一个节点上则拷贝从 1.0s 退化到 6.5s。节点离卡远近无所谓，分开才是关键。
#ifdef __linux__
bool PinTo(const std::string& list) {
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t comma = list.find(',', begin);
        const std::string item = list.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!item.empty()) {
            const size_t dash = item.find('-');
            const int lo = std::atoi(item.c_str());
            const int hi = dash == std::string::npos ? lo : std::atoi(item.c_str() + dash + 1);
            for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c)
                CPU_SET(c, &set);
        }
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    return CPU_COUNT(&set) > 0 && sched_setaffinity(0, sizeof(set), &set) == 0;
}

// /sys 里第 n 个 NUMA 节点的核列表，用来给第 n 张卡分核
std::string NodeCpus(size_t n) {
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%zu/cpulist", n);
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr)
        return "";
    char buf[512] = {0};
    if (std::fgets(buf, sizeof(buf), f) == nullptr)
        buf[0] = '\0';
    std::fclose(f);
    std::string out = buf;
    while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
        out.pop_back();
    return out;
}
#else
bool PinTo(const std::string&) { return false; }
std::string NodeCpus(size_t) { return ""; }
#endif

struct Shard {
    int device = 0;
    std::string cpus;              // 空 = 不绑核
    std::string file;              // 单文件模式；空 = 不读单个文件
    std::vector<Piece> pieces;     // 目录模式：按顺序读这些
    size_t bytes = 0;
    double readMs = 0;
    double h2dMs = 0;
    bool ok = false;
};

// 一张卡的一轮：读盘 -> 拷到设备。两段分别计时，和 fr_search 日志里的 bulk / H2D 对得上。
void Worker(Shard* shard, int chunkMiB, int readThreads) {
    // 先绑核再建读线程：后建的线程继承这里的亲和性
    if (!shard->cpus.empty() && !PinTo(shard->cpus))
        std::fprintf(stderr, "[LoadBench] 卡 %d: 绑核 %s 失败\n", shard->device, shard->cpus.c_str());
    if (aclrtSetDevice(shard->device) != ACL_SUCCESS) {
        std::fprintf(stderr, "[LoadBench] device %d: aclrtSetDevice 失败\n", shard->device);
        return;
    }
    void* host = std::malloc(shard->bytes);
    if (host == nullptr) {
        std::fprintf(stderr, "[LoadBench] device %d: 主机内存分配失败（%zu 字节）\n", shard->device, shard->bytes);
        return;
    }
    void* dev = nullptr;
    if (aclrtMalloc(&dev, shard->bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "[LoadBench] device %d: 显存分配失败（%zu 字节）\n", shard->device, shard->bytes);
        std::free(host);
        return;
    }

    if (!shard->pieces.empty()) {
        // 多线程分文件读：每个 Piece 有自己的落点，线程之间不重叠，用一个原子下标领活。
        // 单线程 copy_to_user 约 0.9 GB/s 到顶，这里主要是为了把它绕开。
        const size_t chunk = static_cast<size_t>(chunkMiB) * 1048576u;
        const auto t0 = Clock::now();
        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0};
        const int nThreads = std::max(1, std::min(readThreads, static_cast<int>(shard->pieces.size())));
        std::vector<std::thread> readers;
        readers.reserve(static_cast<size_t>(nThreads));
        for (int t = 0; t < nThreads; ++t) {
            readers.emplace_back([&] {
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= shard->pieces.size())
                        return;
                    const Piece& p = shard->pieces[i];
                    const int fd = ::open(p.path.c_str(), O_RDONLY);
                    if (fd < 0) {
                        std::fprintf(stderr, "[LoadBench] 卡 %d: 打不开 %s\n", shard->device, p.path.c_str());
                        return;
                    }
                    size_t off = 0;
                    while (off < p.bytes) {
                        const size_t want = std::min(chunk, p.bytes - off);
                        const ssize_t got = ::pread(fd, static_cast<char*>(host) + p.dstOffset + off, want,
                                                    static_cast<off_t>(off));
                        if (got <= 0)
                            break;
                        off += static_cast<size_t>(got);
                    }
                    ::close(fd);
                    done.fetch_add(off);
                }
            });
        }
        for (std::thread& th : readers)
            th.join();
        shard->readMs = MsSince(t0);
        if (done.load() != shard->bytes) {
            std::fprintf(stderr, "[LoadBench] 卡 %d: 只读到 %zu/%zu 字节\n", shard->device, done.load(),
                         shard->bytes);
            aclrtFree(dev);
            std::free(host);
            return;
        }
    } else if (shard->file.empty()) {
        // 数据来自内存：先把页真正摸一遍，否则缺页的代价会算进拷贝里，量到的就不是拷贝
        std::memset(host, 1, shard->bytes);
    } else {
        const int fd = ::open(shard->file.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[LoadBench] device %d: 打不开 %s\n", shard->device, shard->file.c_str());
            aclrtFree(dev);
            std::free(host);
            return;
        }
        const size_t chunk = static_cast<size_t>(chunkMiB) * 1048576u;
        const auto t0 = Clock::now();
        size_t done = 0;
        while (done < shard->bytes) {
            const size_t want = std::min(chunk, shard->bytes - done);
            const ssize_t got = ::pread(fd, static_cast<char*>(host) + done, want, static_cast<off_t>(done));
            if (got <= 0)
                break;
            done += static_cast<size_t>(got);
        }
        shard->readMs = MsSince(t0);
        ::close(fd);
        if (done != shard->bytes) {
            std::fprintf(stderr, "[LoadBench] device %d: 只读到 %zu/%zu 字节\n", shard->device, done, shard->bytes);
            aclrtFree(dev);
            std::free(host);
            return;
        }
    }

    const auto t1 = Clock::now();
    const bool copied =
        aclrtMemcpy(dev, shard->bytes, host, shard->bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    shard->h2dMs = MsSince(t1);
    if (!copied)
        std::fprintf(stderr, "[LoadBench] device %d: aclrtMemcpy 失败\n", shard->device);

    aclrtFree(dev);
    std::free(host);
    shard->ok = copied;
}

}  // namespace

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::vector<std::string> devText = Split(FLAGS_devices);
    const std::vector<std::string> files = Split(FLAGS_files);
    const std::vector<std::string> dirs = Split(FLAGS_dirs);
    if (devText.empty()) {
        std::fprintf(stderr, "[LoadBench] --devices 是空的\n");
        return 1;
    }
    if (!files.empty() && files.size() != devText.size()) {
        std::fprintf(stderr, "[LoadBench] --files 给了 %zu 个，--devices 是 %zu 个，要一一对应\n", files.size(),
                     devText.size());
        return 1;
    }
    if (!dirs.empty() && dirs.size() != devText.size()) {
        std::fprintf(stderr, "[LoadBench] --dirs 给了 %zu 个，--devices 是 %zu 个，要一一对应\n", dirs.size(),
                     devText.size());
        return 1;
    }
    if (!dirs.empty() && !files.empty()) {
        std::fprintf(stderr, "[LoadBench] --dirs 和 --files 只能给一个\n");
        return 1;
    }

    // 每卡的核列表：auto 按 NUMA 节点依次分，空串不绑，否则按 ';' 切
    std::vector<std::string> cpuList;
    if (FLAGS_card_cpus == "auto") {
        for (size_t i = 0; i < devText.size(); ++i)
            cpuList.push_back(NodeCpus(i));
    } else if (!FLAGS_card_cpus.empty()) {
        size_t begin = 0;
        while (begin <= FLAGS_card_cpus.size()) {
            const size_t semi = FLAGS_card_cpus.find(';', begin);
            cpuList.push_back(FLAGS_card_cpus.substr(
                begin, semi == std::string::npos ? std::string::npos : semi - begin));
            if (semi == std::string::npos)
                break;
            begin = semi + 1;
        }
        if (cpuList.size() != devText.size()) {
            std::fprintf(stderr, "[LoadBench] --card_cpus 给了 %zu 份，--devices 是 %zu 个，要一一对应\n",
                         cpuList.size(), devText.size());
            return 1;
        }
    }

    std::vector<Shard> shards(devText.size());
    size_t total = 0;
    for (size_t i = 0; i < devText.size(); ++i) {
        shards[i].device = std::atoi(devText[i].c_str());
        if (i < cpuList.size())
            shards[i].cpus = cpuList[i];
        if (!dirs.empty()) {
            const size_t quota = static_cast<size_t>(FLAGS_gib_per_card * 1073741824.0);
            size_t got = 0;
            // 目录里不够 quota 就读多少算多少：--gib_per_card 是上限不是硬要求，
            // 分片之间本来就不等大，拿 du 的占盘数去对文件实际字节必然差一截。
            shards[i].pieces = TakeFrom(dirs[i], quota, &got);
            if (got == 0) {
                std::fprintf(stderr, "[LoadBench] %s 里没有可读的文件\n", dirs[i].c_str());
                return 1;
            }
            shards[i].bytes = got;
        } else if (!files.empty()) {
            shards[i].file = files[i];
            struct stat st {};
            if (::stat(files[i].c_str(), &st) != 0) {
                std::fprintf(stderr, "[LoadBench] 读不到 %s\n", files[i].c_str());
                return 1;
            }
            shards[i].bytes = static_cast<size_t>(st.st_size);
        } else {
            shards[i].bytes = static_cast<size_t>(FLAGS_gib_per_card * 1073741824.0);
        }
        total += shards[i].bytes;
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "[LoadBench] aclInit 失败\n");
        return 1;
    }
    std::printf("[LoadBench] 共 %.2f GiB，分给 %zu 张卡同时搬，%s，跑 %d 轮\n",
                static_cast<double>(total) / 1073741824.0, shards.size(),
                (files.empty() && dirs.empty()) ? "数据来自内存，只测拷到设备" : "读盘后拷到设备",
                FLAGS_repeat);
    for (const Shard& sh : shards)
        if (!sh.cpus.empty())
            std::printf("            卡 %d 绑核 %s\n", sh.device, sh.cpus.c_str());
    std::fflush(stdout);

    std::vector<double> walls;
    int failed = 0;
    for (int rep = 1; rep <= FLAGS_repeat; ++rep) {
        for (Shard& s : shards) {
            s.readMs = s.h2dMs = 0;
            s.ok = false;
        }
        const auto t0 = Clock::now();
        std::vector<std::thread> pool;
        pool.reserve(shards.size());
        for (Shard& s : shards)
            pool.emplace_back(Worker, &s, FLAGS_read_chunk_mib, FLAGS_read_threads);
        for (std::thread& th : pool)
            th.join();
        const double wall = MsSince(t0);

        // 一轮一行：总耗时在前，各卡的分段跟在后面，不然 3 轮 2 卡就是十几行
        bool allOk = true;
        std::printf("  第 %d 轮  %.2fs (%.2f GiB/s)  ", rep, wall / 1000.0, GiBps(total, wall));
        for (const Shard& s : shards) {
            allOk = allOk && s.ok;
            if (s.file.empty() && s.pieces.empty())
                std::printf("  卡%d 拷 %.2f", s.device, s.h2dMs / 1000.0);
            else
                std::printf("  卡%d 读 %.2f 拷 %.2f", s.device, s.readMs / 1000.0, s.h2dMs / 1000.0);
        }
        std::printf("\n");
        std::fflush(stdout);
        if (allOk)
            walls.push_back(wall);
        else
            ++failed;
    }

    if (!walls.empty()) {
        std::vector<double> sorted = walls;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        std::printf("[LoadBench] 中位数 %.2fs (%.2f GiB/s)\n", median / 1000.0, GiBps(total, median));
    }
    if (failed > 0)
        std::fprintf(stderr, "[LoadBench] 有 %d 轮没跑成\n", failed);

    aclFinalize();
    return failed > 0 ? 1 : 0;
}
