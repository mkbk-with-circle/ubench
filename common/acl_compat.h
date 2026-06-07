#pragma once

#if defined(UBENCH_STUB_ACL)

#include <cstddef>
#include <cstdint>

using aclError = int;
using aclrtContext = void*;
using aclrtStream = void*;

constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;

inline aclError aclInit(const char*) { return ACL_SUCCESS; }
inline aclError aclFinalize() { return ACL_SUCCESS; }
inline aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtResetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtCreateContext(aclrtContext* ctx, int32_t) {
  *ctx = reinterpret_cast<void*>(0x1);
  return ACL_SUCCESS;
}
inline aclError aclrtDestroyContext(aclrtContext) { return ACL_SUCCESS; }
inline aclError aclrtCreateStream(aclrtStream* stream) {
  *stream = reinterpret_cast<void*>(0x1);
  return ACL_SUCCESS;
}
inline aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
inline aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }
inline aclError aclrtMalloc(void** ptr, size_t bytes, int) {
  *ptr = ::operator new(bytes);
  return ACL_SUCCESS;
}
inline aclError aclrtFree(void* ptr) {
  ::operator delete(ptr);
  return ACL_SUCCESS;
}
inline aclError aclrtMemcpy(void* dst, size_t, const void* src, size_t count, int) {
  const auto* s = static_cast<const uint8_t*>(src);
  auto* d = static_cast<uint8_t*>(dst);
  for (size_t i = 0; i < count; ++i) {
    d[i] = s[i];
  }
  return ACL_SUCCESS;
}
inline aclError aclrtMemset(void* dst, size_t, int value, size_t count) {
  auto* d = static_cast<uint8_t*>(dst);
  for (size_t i = 0; i < count; ++i) {
    d[i] = static_cast<uint8_t>(value);
  }
  return ACL_SUCCESS;
}

#else
#include <acl/acl.h>
#endif
