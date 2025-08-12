#pragma once

#include <memory>
#include <set>
#include <unordered_map>

#include <torch/extension.h>

#include "constants.hpp"
#include "page.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class FTensor {
public:
  FTensor(const std::string &name, size_t size, torch::Dtype dtype,
          torch::Device dev, std::shared_ptr<Page> zero_page);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline torch::Tensor get_tensor() noexcept { return tensor_; }

  // [GVM] swap out `size` bytes from the tensor.
  bool reclaim_handler(size_t size);

  // [GVM] call UVM prefetch to host or internal swap interface.
  bool swapout(void *addr, size_t size);

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  torch::Dtype dtype_;
  torch::Device dev_;
  std::shared_ptr<Page> zero_page_;

  torch::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapped_pages_;
  std::set<page_id_t> unmapped_pages_;
};

} // namespace kvcached
