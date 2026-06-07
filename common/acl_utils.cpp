#include "acl_utils.h"

#include <chrono>

namespace ubench {

AclRuntime::AclRuntime(int32_t device_id) : device_id_(device_id) {
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(device_id_));
  CHECK_ACL(aclrtCreateContext(&context_, device_id_));
  CHECK_ACL(aclrtCreateStream(&stream_));
}

AclRuntime::~AclRuntime() {
  if (stream_ != nullptr) {
    (void)aclrtDestroyStream(stream_);
  }
  if (context_ != nullptr) {
    (void)aclrtDestroyContext(context_);
  }
  (void)aclrtResetDevice(device_id_);
  (void)aclFinalize();
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  CHECK_ACL(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_ != nullptr) {
    (void)aclrtFree(ptr_);
  }
}

double NowMicros() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) /
      1000.0;
}

}  // namespace ubench
