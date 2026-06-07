// Wrapper for __cce_rtKernelLaunch - bridges bisheng host compilation to CANN runtime
#include <cstdint>
#include <cstring>

// Forward declarations from CANN runtime
extern "C" {
typedef int rtError_t;
typedef void* rtStream_t;

struct rtSmDesc_t;

rtError_t rtKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args,
                         uint32_t argsSize, rtSmDesc_t* smDesc, rtStream_t stream);
}

// __cce_rtKernelLaunch is called by bisheng-compiled host code
// Arguments based on disassembly analysis:
//   x0: stubFunc (kernel handle/pointer)
//   x1: blockDim
//   x2: args (pointer to args on stack)
//   x3: argsSize
//   x4: smDesc (or stream?)
//   x5: stream (or smDesc?)
//   x6: kernelName (unused for launch)
//   x7: kernelId (unused for launch)
extern "C" int64_t __cce_rtKernelLaunch(const void* stubFunc, uint32_t blockDim,
                                        void* args, uint32_t argsSize,
                                        void* p4, void* p5,
                                        const char* kernelName, uint32_t kernelId) {
    // Based on the calling convention, p4 is smDesc and p5 is stream
    return rtKernelLaunch(stubFunc, blockDim, args, argsSize,
                          reinterpret_cast<rtSmDesc_t*>(p4),
                          reinterpret_cast<rtStream_t>(p5));
}
