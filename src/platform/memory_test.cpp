// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region and its undoing, available_ram
// (CPU only, no GPU, no model).  On Linux a lock that ulimit -l (RLIMIT_MEMLOCK) does not allow is skipped, not
// failed (exit 77): the region cannot be locked there (8 MiB on CI's runners).
#include "lamina/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#endif

int main() {
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const lamina::platform::LockResult r = lamina::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    bool skipped = false;
#if !defined(_WIN32)
    rlimit rl{};
    if (!r.ok && getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < bytes) {
        std::printf("lock_resident: SKIPPED, ulimit -l is %llu KiB, below the region's 256 MiB\n",
                    (unsigned long long) (rl.rlim_cur >> 10));
        skipped = true;
    }
#endif
    int fail = !skipped && !(r.ok && r.locked_bytes == bytes);
    // the conversation cache's RAM reserve (--ram-reserve-mib) reads this figure
    uint64_t avail = 0;
    const bool got = lamina::platform::available_ram(avail);
    std::printf("available_ram: ok=%d %llu MiB\n", (int) got, (unsigned long long) (avail >> 20));
    fail |= !got || avail == 0;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    std::printf("working set %llu MiB\n", (unsigned long long) (pmc.WorkingSetSize >> 20));
    fail |= pmc.WorkingSetSize < bytes;
#endif
    lamina::platform::unlock_resident(p, bytes);
#if defined(_WIN32)
    // the working-set minimum the lock raised is lowered by its unlock: three more rounds leave it where it was
    SIZE_T min0 = 0, max0 = 0, min1 = 0, max1 = 0;
    DWORD flags = 0;
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min0, &max0, &flags);
    for (int k = 0; k < 3; ++k) {
        const lamina::platform::LockResult again = lamina::platform::lock_resident(p, bytes);
        lamina::platform::unlock_resident(p, again.locked_bytes);
    }
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min1, &max1, &flags);
    std::printf("working-set minimum %llu MiB, after three more locks and unlocks %llu MiB\n",
                (unsigned long long) (min0 >> 20), (unsigned long long) (min1 >> 20));
    fail |= min1 != min0;
#endif
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : skipped ? "OK, the lock SKIPPED" : "OK");
    return fail ? 1 : skipped ? 77 : 0;
}
