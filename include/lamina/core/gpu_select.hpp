// include/lamina/core/gpu_select.hpp - GPUs named in the options, a short probe that orders and sizes the GPU tiers,
// and a tier's clocks locked while the engine generates.
//
// A device ordinal moves when a card is added: here an RTX 5060 Ti on an M.2 adapter took bus 02 and made the RTX
// 3090 device 2 instead of 1, and its slot's bus 0F became 10, so a PCI bus id moves too.  A UUID stays with its
// card, and so does a name while no other card has it: --main-gpu and --second-gpu take any of the four.
#pragma once

#include <string>
#include <vector>

namespace lamina::core {

/// One visible CUDA device, as the options can name it (no context is made to list it).
struct GpuInfo {
    int ordinal = -1;
    std::string name;     ///< cudaDeviceProp::name, "NVIDIA GeForce RTX 3090"
    std::string bus_id;   ///< "0000:10:00.0"
    std::string uuid;     ///< "GPU-f26e9ba6-34e6-e46a-a700-b8349f7e5456", as nvidia-smi prints it
    int sms = 0;
};
/// Every visible device, its ordinal its index.
std::vector<GpuInfo> list_gpus();

/// The ordinal `spec` names: an ordinal ("2"), a UUID or a prefix of it that only one card has ("GPU-f26e9ba6"), a PCI
/// bus id ("10:00.0", "0000:10:00.0", nvidia-smi's "00000000:10:00.0"), or a name or a part of one that only one card
/// has ("RTX 3090", case-insensitive; a whole name wins over a part of another; a number no ordinal is, as a word of
/// one: "3090", not "7").  -1
/// with `why` when none or several match (then each one's ordinal, name and UUID).  serve/server.py's find_gpu reads
/// names the same way.
int find_gpu(const std::vector<GpuInfo>& gpus, const std::string& spec, std::string& why);

/// A tier's GPU as a short probe saw it: the pinned host-to-device rate of 64 MiB copies (GB/s, the best of five
/// after the link is woken, and of five more for as long as those were faster), and whether the desktop or another
/// graphics program uses it (NVML: a display active or a process's graphics context; unknown when NVML is not there:
/// true, the larger reserve).
struct GpuProbe {
    double h2d_gbps = 0;
    bool display = true;
    bool display_known = false;
};
/// The devices' probes: their links woken together (copies until each link is at its full PCIe generation, 0.4-0.5 s
/// here), then each one's copies timed alone; `main_device` current after.
std::vector<GpuProbe> probe_gpus(const std::vector<int>& devices, int main_device);

/// GPUs' clocks locked while the engine generates (--second-gpu-clocks): under a tier's short shares the driver lowers
/// that card's clocks between them, and its rows come late; locked all the time it draws its awake power while the
/// engine idles or reads a prompt.  The locks are NVML's (nvidia-smi's -lgc and -lmc: they need administrator rights
/// and stay until released or a reboot), set and released on a thread of this object's: a lock is at speed about
/// half a second after it is asked for.
class GpuClocks {
public:
    GpuClocks() = default;
    ~GpuClocks();   ///< releases the locks it holds
    GpuClocks(const GpuClocks&) = delete;
    GpuClocks& operator=(const GpuClocks&) = delete;

    /// `spec` as "<graphics min>-<graphics max>/<memory>" MHz ("1395-2100/9751"; a memory clock of 0 leaves that
    /// one to the driver): that lock for `device` (a CUDA ordinal).  False with `why` for another form; values
    /// outside the card's range are NVML's to judge, and it takes them without a word (above the range: the card's
    /// top clocks; below it: no lock).  Before the first hold().
    bool add(int device, const std::string& spec, std::string& why);
    /// Generation nears or runs: the locks are set unless they are, and a release that was coming is off.  Returns at
    /// once.
    void hold();
    /// Generation is over: the locks are released `seconds` from now unless hold() comes first.
    void release_after(double seconds);
    /// The locks of the object that held some, released on the calling thread: for an exit that runs no destructor
    /// (std::exit runs it by itself; std::_Exit does not).
    static void release_now();

private:
    struct Gpu {
        int device = -1;
        void* nvml = nullptr;
        unsigned gr_min = 0, gr_max = 0, mem = 0;
    };
    struct State;
    void run();
    bool set(bool lock);
    std::vector<Gpu> gpus_;
    State* st_ = nullptr;
};

}  // namespace lamina::core
