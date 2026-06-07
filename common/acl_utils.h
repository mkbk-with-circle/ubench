#pragma once

#include "acl_compat.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#define CHECK_ACL(expr)                                                        \
  do {                                                                         \
    aclError _acl_ret = (expr);                                                \
    if (_acl_ret != ACL_SUCCESS) {                                             \
      throw std::runtime_error(std::string("ACL call failed: ") + #expr +      \
                               " error=" + std::to_string(_acl_ret));         \
    }                                                                          \
  } while (0)

namespace ubench {

class AclRuntime {
 public:
  explicit AclRuntime(int32_t device_id);
  ~AclRuntime();

  AclRuntime(const AclRuntime&) = delete;
  AclRuntime& operator=(const AclRuntime&) = delete;

  aclrtStream stream() const { return stream_; }
  int32_t device_id() const { return device_id_; }

 private:
  int32_t device_id_;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

double NowMicros();

}  // namespace ubench
