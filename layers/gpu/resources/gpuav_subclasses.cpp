/* Copyright (c) 2018-2024 The Khronos Group Inc.
 * Copyright (c) 2018-2024 Valve Corporation
 * Copyright (c) 2018-2024 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu/resources/gpuav_subclasses.h"

#include "gpu/core/gpuav.h"
#include "gpu/core/gpuav_constants.h"
#include "gpu/descriptor_validation/gpuav_image_layout.h"
#include "gpu/error_message/gpuav_vuids.h"
#include "gpu/descriptor_validation/gpuav_descriptor_validation.h"
#include "gpu/shaders/gpu_error_header.h"
#include "state_tracker/shader_object_state.h"

namespace gpuav {

Buffer::Buffer(ValidationStateTracker &dev_data, VkBuffer buff, const VkBufferCreateInfo *pCreateInfo, DescriptorHeap &desc_heap_)
    : vvl::Buffer(dev_data, buff, pCreateInfo),
      desc_heap(desc_heap_),
      id(desc_heap.NextId(VulkanTypedHandle(buff, kVulkanObjectTypeBuffer))) {}

void Buffer::Destroy() {
    desc_heap.DeleteId(id);
    vvl::Buffer::Destroy();
}

void Buffer::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::Buffer::NotifyInvalidate(invalid_nodes, unlink);
}

BufferView::BufferView(const std::shared_ptr<vvl::Buffer> &bf, VkBufferView bv, const VkBufferViewCreateInfo *ci,
                       VkFormatFeatureFlags2KHR buf_ff, DescriptorHeap &desc_heap_)
    : vvl::BufferView(bf, bv, ci, buf_ff),
      desc_heap(desc_heap_),
      id(desc_heap.NextId(VulkanTypedHandle(bv, kVulkanObjectTypeBufferView))) {}

void BufferView::Destroy() {
    desc_heap.DeleteId(id);
    vvl::BufferView::Destroy();
}

void BufferView::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::BufferView::NotifyInvalidate(invalid_nodes, unlink);
}

ImageView::ImageView(const std::shared_ptr<vvl::Image> &image_state, VkImageView iv, const VkImageViewCreateInfo *ci,
                     VkFormatFeatureFlags2KHR ff, const VkFilterCubicImageViewImageFormatPropertiesEXT &cubic_props,
                     DescriptorHeap &desc_heap_)
    : vvl::ImageView(image_state, iv, ci, ff, cubic_props),
      desc_heap(desc_heap_),
      id(desc_heap.NextId(VulkanTypedHandle(iv, kVulkanObjectTypeImageView))) {}

void ImageView::Destroy() {
    desc_heap.DeleteId(id);
    vvl::ImageView::Destroy();
}

void ImageView::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::ImageView::NotifyInvalidate(invalid_nodes, unlink);
}

Sampler::Sampler(const VkSampler s, const VkSamplerCreateInfo *pci, DescriptorHeap &desc_heap_)
    : vvl::Sampler(s, pci), desc_heap(desc_heap_), id(desc_heap.NextId(VulkanTypedHandle(s, kVulkanObjectTypeSampler))) {}

void Sampler::Destroy() {
    desc_heap.DeleteId(id);
    vvl::Sampler::Destroy();
}

void Sampler::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::Sampler::NotifyInvalidate(invalid_nodes, unlink);
}

AccelerationStructureKHR::AccelerationStructureKHR(VkAccelerationStructureKHR as, const VkAccelerationStructureCreateInfoKHR *ci,
                                                   std::shared_ptr<vvl::Buffer> &&buf_state, DescriptorHeap &desc_heap_)
    : vvl::AccelerationStructureKHR(as, ci, std::move(buf_state)),
      desc_heap(desc_heap_),
      id(desc_heap.NextId(VulkanTypedHandle(as, kVulkanObjectTypeAccelerationStructureKHR))) {}

void AccelerationStructureKHR::Destroy() {
    desc_heap.DeleteId(id);
    vvl::AccelerationStructureKHR::Destroy();
}

void AccelerationStructureKHR::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::AccelerationStructureKHR::NotifyInvalidate(invalid_nodes, unlink);
}

AccelerationStructureNV::AccelerationStructureNV(VkDevice device, VkAccelerationStructureNV as,
                                                 const VkAccelerationStructureCreateInfoNV *ci, DescriptorHeap &desc_heap_)
    : vvl::AccelerationStructureNV(device, as, ci),
      desc_heap(desc_heap_),
      id(desc_heap.NextId(VulkanTypedHandle(as, kVulkanObjectTypeAccelerationStructureNV))) {}

void AccelerationStructureNV::Destroy() {
    desc_heap.DeleteId(id);
    vvl::AccelerationStructureNV::Destroy();
}

void AccelerationStructureNV::NotifyInvalidate(const NodeList &invalid_nodes, bool unlink) {
    desc_heap.DeleteId(id);
    vvl::AccelerationStructureNV::NotifyInvalidate(invalid_nodes, unlink);
}

CommandBuffer::CommandBuffer(Validator &gpuav, VkCommandBuffer handle, const VkCommandBufferAllocateInfo *pCreateInfo,
                             const vvl::CommandPool *pool)
    : gpu_tracker::CommandBuffer(gpuav, handle, pCreateInfo, pool),
      gpu_resources_manager(gpuav.vma_allocator_, *gpuav.desc_set_manager_),
      state_(gpuav) {
    AllocateResources();
}

static bool AllocateErrorLogsBuffer(Validator &gpuav, gpu::DeviceMemoryBlock &error_logs_mem, const Location &loc) {
    VkBufferCreateInfo buffer_info = vku::InitStructHelper();
    buffer_info.size = glsl::kErrorBufferByteSize;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    alloc_info.pool = gpuav.output_buffer_pool_;
    VkResult result = vmaCreateBuffer(gpuav.vma_allocator_, &buffer_info, &alloc_info, &error_logs_mem.buffer,
                                      &error_logs_mem.allocation, nullptr);
    if (result != VK_SUCCESS) {
        gpuav.InternalError(gpuav.device, loc, "Unable to allocate device memory for error output buffer. Aborting GPU-AV.", true);
        return false;
    }

    uint32_t *output_buffer_ptr;
    result = vmaMapMemory(gpuav.vma_allocator_, error_logs_mem.allocation, reinterpret_cast<void **>(&output_buffer_ptr));
    if (result == VK_SUCCESS) {
        memset(output_buffer_ptr, 0, glsl::kErrorBufferByteSize);
        if (gpuav.gpuav_settings.validate_descriptors) {
            output_buffer_ptr[cst::stream_output_flags_offset] = cst::inst_buffer_oob_enabled;
        }
        vmaUnmapMemory(gpuav.vma_allocator_, error_logs_mem.allocation);
    } else {
        gpuav.InternalError(gpuav.device, loc, "Unable to map device memory allocated for error output buffer. Aborting GPU-AV.",
                            true);
        return false;
    }

    return true;
}

void CommandBuffer::AllocateResources() {
    using Func = vvl::Func;

    auto gpuav = static_cast<Validator *>(&dev_data);

    VkResult result = VK_SUCCESS;

    // Instrumentation descriptor set layout
    if (instrumentation_desc_set_layout_ == VK_NULL_HANDLE) {
        assert(!gpuav->instrumentation_bindings_.empty());
        VkDescriptorSetLayoutCreateInfo instrumentation_desc_set_layout_ci = vku::InitStructHelper();
        instrumentation_desc_set_layout_ci.bindingCount = static_cast<uint32_t>(gpuav->instrumentation_bindings_.size());
        instrumentation_desc_set_layout_ci.pBindings = gpuav->instrumentation_bindings_.data();
        result = DispatchCreateDescriptorSetLayout(gpuav->device, &instrumentation_desc_set_layout_ci, nullptr,
                                                   &instrumentation_desc_set_layout_);
        if (result != VK_SUCCESS) {
            gpuav->InternalError(gpuav->device, Location(Func::vkAllocateCommandBuffers),
                                 "Unable to create instrumentation descriptor set layout. Aborting GPU-AV.");
            return;
        }
    }

    // Error output buffer
    if (!AllocateErrorLogsBuffer(*gpuav, error_output_buffer_, Location(Func::vkAllocateCommandBuffers))) {
        return;
    }

    // Commands errors counts buffer
    {
        VkBufferCreateInfo buffer_info = vku::InitStructHelper();
        buffer_info.size = GetCmdErrorsCountsBufferByteSize();
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_info.pool = gpuav->output_buffer_pool_;
        result = vmaCreateBuffer(gpuav->vma_allocator_, &buffer_info, &alloc_info, &cmd_errors_counts_buffer_.buffer,
                                 &cmd_errors_counts_buffer_.allocation, nullptr);
        if (result != VK_SUCCESS) {
            gpuav->InternalError(gpuav->device, Location(Func::vkAllocateCommandBuffers),
                                 "Unable to allocate device memory for commands errors counts buffer. Aborting GPU-AV.", true);
            return;
        }

        ClearCmdErrorsCountsBuffer();
        if (gpuav->aborted_) return;
    }

    // BDA snapshot
    if (gpuav->gpuav_settings.validate_bda) {
        VkBufferCreateInfo buffer_info = vku::InitStructHelper();
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VmaAllocationCreateInfo alloc_info = {};
        buffer_info.size = GetBdaRangesBufferByteSize();
        // This buffer could be very large if an application uses many buffers. Allocating it as HOST_CACHED
        // and manually flushing it at the end of the state updates is faster than using HOST_COHERENT.
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        result = vmaCreateBuffer(gpuav->vma_allocator_, &buffer_info, &alloc_info, &bda_ranges_snapshot_.buffer,
                                 &bda_ranges_snapshot_.allocation, nullptr);
        if (result != VK_SUCCESS) {
            gpuav->InternalError(gpuav->device, Location(Func::vkAllocateCommandBuffers),
                                 "Unable to allocate device memory for buffer device address data. Aborting GPU-AV.", true);
            return;
        }
    }

    // Update validation commands common descriptor set
    {
        const std::vector<VkDescriptorSetLayoutBinding> validation_cmd_bindings = {
            // Error output buffer
            {glsl::kBindingDiagErrorBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
            // Buffer holding action command index in command buffer
            {glsl::kBindingDiagActionIndex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr},
            // Buffer holding a resource index from the per command buffer command resources list
            {glsl::kBindingDiagCmdResourceIndex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr},
            // Commands errors counts buffer
            {glsl::kBindingDiagCmdErrorsCount, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        };

        if (validation_cmd_desc_set_layout_ == VK_NULL_HANDLE) {
            VkDescriptorSetLayoutCreateInfo validation_cmd_desc_set_layout_ci = vku::InitStructHelper();
            validation_cmd_desc_set_layout_ci.bindingCount = static_cast<uint32_t>(validation_cmd_bindings.size());
            validation_cmd_desc_set_layout_ci.pBindings = validation_cmd_bindings.data();
            result = DispatchCreateDescriptorSetLayout(gpuav->device, &validation_cmd_desc_set_layout_ci, nullptr,
                                                       &validation_cmd_desc_set_layout_);
            if (result != VK_SUCCESS) {
                gpuav->InternalError(gpuav->device, Location(Func::vkAllocateCommandBuffers),
                                     "Unable to create descriptor set layout used for validation commands. Aborting GPU-AV.");
                return;
            }
        }

        assert(validation_cmd_desc_pool_ == VK_NULL_HANDLE);
        assert(validation_cmd_desc_set_ == VK_NULL_HANDLE);
        result = gpuav->desc_set_manager_->GetDescriptorSet(&validation_cmd_desc_pool_, validation_cmd_desc_set_layout_,
                                                            &validation_cmd_desc_set_);
        if (result != VK_SUCCESS) {
            gpuav->InternalError(gpuav->device, Location(Func::vkAllocateCommandBuffers),
                                 "Unable to create descriptor set used for validation commands. Aborting GPU-AV.");
            return;
        }

        std::array<VkWriteDescriptorSet, 4> validation_cmd_descriptor_writes = {};
        assert(validation_cmd_bindings.size() == validation_cmd_descriptor_writes.size());

        VkDescriptorBufferInfo error_output_buffer_desc_info = {};

        assert(error_output_buffer_.buffer != VK_NULL_HANDLE);
        error_output_buffer_desc_info.buffer = error_output_buffer_.buffer;
        error_output_buffer_desc_info.offset = 0;
        error_output_buffer_desc_info.range = VK_WHOLE_SIZE;

        validation_cmd_descriptor_writes[0] = vku::InitStructHelper();
        validation_cmd_descriptor_writes[0].dstBinding = glsl::kBindingDiagErrorBuffer;
        validation_cmd_descriptor_writes[0].descriptorCount = 1;
        validation_cmd_descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        validation_cmd_descriptor_writes[0].pBufferInfo = &error_output_buffer_desc_info;
        validation_cmd_descriptor_writes[0].dstSet = GetValidationCmdCommonDescriptorSet();

        VkDescriptorBufferInfo cmd_indices_buffer_desc_info = {};

        assert(error_output_buffer_.buffer != VK_NULL_HANDLE);
        cmd_indices_buffer_desc_info.buffer = gpuav->indices_buffer_.buffer;
        cmd_indices_buffer_desc_info.offset = 0;
        cmd_indices_buffer_desc_info.range = sizeof(uint32_t);

        validation_cmd_descriptor_writes[1] = vku::InitStructHelper();
        validation_cmd_descriptor_writes[1].dstBinding = glsl::kBindingDiagActionIndex;
        validation_cmd_descriptor_writes[1].descriptorCount = 1;
        validation_cmd_descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        validation_cmd_descriptor_writes[1].pBufferInfo = &cmd_indices_buffer_desc_info;
        validation_cmd_descriptor_writes[1].dstSet = GetValidationCmdCommonDescriptorSet();

        validation_cmd_descriptor_writes[2] = validation_cmd_descriptor_writes[1];
        validation_cmd_descriptor_writes[2].dstBinding = glsl::kBindingDiagCmdResourceIndex;

        VkDescriptorBufferInfo cmd_errors_count_buffer_desc_info = {};
        cmd_errors_count_buffer_desc_info.buffer = GetCmdErrorsCountsBuffer();
        cmd_errors_count_buffer_desc_info.offset = 0;
        cmd_errors_count_buffer_desc_info.range = VK_WHOLE_SIZE;

        validation_cmd_descriptor_writes[3] = vku::InitStructHelper();
        validation_cmd_descriptor_writes[3].dstBinding = glsl::kBindingDiagCmdErrorsCount;
        validation_cmd_descriptor_writes[3].descriptorCount = 1;
        validation_cmd_descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        validation_cmd_descriptor_writes[3].pBufferInfo = &cmd_errors_count_buffer_desc_info;
        validation_cmd_descriptor_writes[3].dstSet = GetValidationCmdCommonDescriptorSet();

        DispatchUpdateDescriptorSets(gpuav->device, static_cast<uint32_t>(validation_cmd_descriptor_writes.size()),
                                     validation_cmd_descriptor_writes.data(), 0, NULL);
    }
}

bool CommandBuffer::UpdateBufferDeviceAddressRangesBuffer() {
    auto gpuav = static_cast<Validator *>(&dev_data);

    // By supplying a "date"
    if (!gpuav->gpuav_settings.validate_bda || bda_ranges_snapshot_version_ == gpuav->buffer_device_address_ranges_version) {
        return true;
    }

    // Update buffer device address table
    // ---
    VkDeviceAddress *bda_table_ptr = nullptr;
    assert(bda_ranges_snapshot_.allocation);
    VkResult result =
        vmaMapMemory(gpuav->vma_allocator_, bda_ranges_snapshot_.allocation, reinterpret_cast<void **>(&bda_table_ptr));
    assert(result == VK_SUCCESS);
    if (result != VK_SUCCESS) {
        if (result != VK_SUCCESS) {
            gpuav->InternalError(gpuav->device, Location(vvl::Func::vkQueueSubmit),
                                 "Unable to map device memory in UpdateBufferDeviceAddressRangesBuffer. Aborting GPU-AV.", true);
            return false;
        }
    }

    // Buffer device address table layout
    // Ranges are sorted from low to high, and do not overlap
    // QWord 0 | Number of *ranges* (1 range occupies 2 QWords)
    // QWord 1 | Range 1 begin
    // QWord 2 | Range 1 end
    // QWord 3 | Range 2 begin
    // QWord 4 | Range 2 end
    // QWord 5 | ...

    const size_t max_recordable_ranges =
        static_cast<size_t>((GetBdaRangesBufferByteSize() - sizeof(uint64_t)) / (2 * sizeof(VkDeviceAddress)));
    auto bda_ranges = reinterpret_cast<ValidationStateTracker::BufferAddressRange *>(bda_table_ptr + 1);
    const auto [ranges_to_update_count, total_address_ranges_count] =
        gpuav->GetBufferAddressRanges(bda_ranges, max_recordable_ranges);
    bda_table_ptr[0] = ranges_to_update_count;

    if (total_address_ranges_count > size_t(gpuav->gpuav_settings.max_bda_in_use)) {
        std::ostringstream problem_string;
        problem_string << "Number of buffer device addresses ranges in use (" << total_address_ranges_count
                       << ") is greater than khronos_validation.gpuav_max_buffer_device_addresses ("
                       << gpuav->gpuav_settings.max_bda_in_use
                       << "). Truncating buffer device address table could result in invalid validation. Aborting GPU-AV.";
        gpuav->InternalError(gpuav->device, Location(vvl::Func::vkQueueSubmit), problem_string.str().c_str());
        return false;
    }

    // Post update cleanups
    // ---
    // Flush the BDA buffer before un-mapping so that the new state is visible to the GPU
    result = vmaFlushAllocation(gpuav->vma_allocator_, bda_ranges_snapshot_.allocation, 0, VK_WHOLE_SIZE);
    vmaUnmapMemory(gpuav->vma_allocator_, bda_ranges_snapshot_.allocation);
    bda_ranges_snapshot_version_ = gpuav->buffer_device_address_ranges_version;

    return true;
}

VkDeviceSize CommandBuffer::GetBdaRangesBufferByteSize() const {
    auto gpuav = static_cast<Validator *>(&dev_data);
    return (1                                           // 1 QWORD for the number of address ranges
            + 2 * gpuav->gpuav_settings.max_bda_in_use  // 2 QWORDS per address range
            ) *
           8;
}

CommandBuffer::~CommandBuffer() { Destroy(); }

void CommandBuffer::Destroy() {
    {
        auto guard = WriteLock();
        ResetCBState();
    }
    vvl::CommandBuffer::Destroy();
}

void CommandBuffer::Reset() {
    vvl::CommandBuffer::Reset();
    ResetCBState();
    // TODO: Calling AllocateResources in Reset like so is a kind of a hack,
    // relying on CommandBuffer internal logic to work.
    // Tried to call it in ResetCBState, hang on command buffer mutex :/
    AllocateResources();
}

void CommandBuffer::ResetCBState() {
    auto gpuav = static_cast<Validator *>(&dev_data);
    // Free the device memory and descriptor set(s) associated with a command buffer.

    gpu_resources_manager.DestroyResources();
    per_command_error_loggers.clear();

    for (auto &buffer_info : di_input_buffer_list) {
        vmaDestroyBuffer(gpuav->vma_allocator_, buffer_info.bindless_state_buffer, buffer_info.bindless_state_buffer_allocation);
    }
    di_input_buffer_list.clear();
    current_bindless_buffer = VK_NULL_HANDLE;

    error_output_buffer_.Destroy(gpuav->vma_allocator_);
    cmd_errors_counts_buffer_.Destroy(gpuav->vma_allocator_);
    bda_ranges_snapshot_.Destroy(gpuav->vma_allocator_);
    bda_ranges_snapshot_version_ = 0;

    if (validation_cmd_desc_pool_ != VK_NULL_HANDLE && validation_cmd_desc_set_ != VK_NULL_HANDLE) {
        gpuav->desc_set_manager_->PutBackDescriptorSet(validation_cmd_desc_pool_, validation_cmd_desc_set_);
        validation_cmd_desc_pool_ = VK_NULL_HANDLE;
        validation_cmd_desc_set_ = VK_NULL_HANDLE;
    }

    if (instrumentation_desc_set_layout_ != VK_NULL_HANDLE) {
        DispatchDestroyDescriptorSetLayout(gpuav->device, instrumentation_desc_set_layout_, nullptr);
        instrumentation_desc_set_layout_ = VK_NULL_HANDLE;
    }

    if (validation_cmd_desc_set_layout_ != VK_NULL_HANDLE) {
        DispatchDestroyDescriptorSetLayout(gpuav->device, validation_cmd_desc_set_layout_, nullptr);
        validation_cmd_desc_set_layout_ = VK_NULL_HANDLE;
    }

    draw_index = 0;
    compute_index = 0;
    trace_rays_index = 0;
}

void CommandBuffer::ClearCmdErrorsCountsBuffer() const {
    auto gpuav = static_cast<Validator *>(&dev_data);
    uint32_t *cmd_errors_counts_buffer_ptr = nullptr;
    VkResult result = vmaMapMemory(gpuav->vma_allocator_, cmd_errors_counts_buffer_.allocation,
                                   reinterpret_cast<void **>(&cmd_errors_counts_buffer_ptr));
    if (result != VK_SUCCESS) {
        gpuav->InternalError(gpuav->device, Location(vvl::Func::vkAllocateCommandBuffers),
                             "Unable to map device memory for commands errors counts buffer. Aborting GPU-AV.", true);
        return;
    }
    std::memset(cmd_errors_counts_buffer_ptr, 0, static_cast<size_t>(GetCmdErrorsCountsBufferByteSize()));
    vmaUnmapMemory(gpuav->vma_allocator_, cmd_errors_counts_buffer_.allocation);
}

bool CommandBuffer::PreProcess() {
    auto gpuav = static_cast<Validator *>(&dev_data);

    bool succeeded = UpdateBindlessStateBuffer(*gpuav, *this, state_.vma_allocator_);
    if (!succeeded) {
        return false;
    }

    succeeded = UpdateBufferDeviceAddressRangesBuffer();
    if (!succeeded) {
        return false;
    }

    return !per_command_error_loggers.empty() || has_build_as_cmd;
}

bool CommandBuffer::NeedsPostProcess() { return !error_output_buffer_.IsNull(); }

// For the given command buffer, map its debug data buffers and read their contents for analysis.
void CommandBuffer::PostProcess(VkQueue queue, const Location &loc) {
    // CommandBuffer::Destroy can happen on an other thread,
    // so when getting here after acquiring command buffer's lock,
    // make sure there are still things to process
    if (!NeedsPostProcess()) {
        return;
    }

    auto gpuav = static_cast<Validator *>(&dev_data);
    bool skip = false;
    uint32_t *error_output_buffer_ptr = nullptr;
    VkResult result =
        vmaMapMemory(gpuav->vma_allocator_, error_output_buffer_.allocation, reinterpret_cast<void **>(&error_output_buffer_ptr));
    assert(result == VK_SUCCESS);
    if (result == VK_SUCCESS) {
        // The second word in the debug output buffer is the number of words that would have
        // been written by the shader instrumentation, if there was enough room in the buffer we provided.
        // The number of words actually written by the shaders is determined by the size of the buffer
        // we provide via the descriptor. So, we process only the number of words that can fit in the
        // buffer.
        const uint32_t total_words = error_output_buffer_ptr[cst::stream_output_size_offset];
        // A zero here means that the shader instrumentation didn't write anything.
        if (total_words != 0) {
            uint32_t *const error_records_start = &error_output_buffer_ptr[cst::stream_output_data_offset];
            assert(glsl::kErrorBufferByteSize > cst::stream_output_data_offset);
            uint32_t *const error_records_end =
                error_output_buffer_ptr + (glsl::kErrorBufferByteSize - cst::stream_output_data_offset);

            uint32_t *error_record_ptr = error_records_start;
            uint32_t record_size = error_record_ptr[glsl::kHeaderErrorRecordSizeOffset];
            assert(record_size == glsl::kErrorRecordSize);

            while (record_size > 0 && (error_record_ptr + record_size) <= error_records_end) {
                const uint32_t error_logger_i = error_record_ptr[glsl::kHeaderCommandResourceIdOffset];
                assert(error_logger_i < per_command_error_loggers.size());
                auto &error_logger = per_command_error_loggers[error_logger_i];
                const LogObjectList objlist(queue, VkHandle());
                skip |= error_logger(*gpuav, error_record_ptr, objlist);

                // Next record
                error_record_ptr += record_size;
                record_size = error_record_ptr[glsl::kHeaderErrorRecordSizeOffset];
            }

            // Clear the written size and any error messages. Note that this preserves the first word, which contains flags.
            assert(glsl::kErrorBufferByteSize > cst::stream_output_data_offset);
            memset(&error_output_buffer_ptr[cst::stream_output_data_offset], 0,
                   glsl::kErrorBufferByteSize - cst::stream_output_data_offset * sizeof(uint32_t));
        }
        error_output_buffer_ptr[cst::stream_output_size_offset] = 0;
        vmaUnmapMemory(gpuav->vma_allocator_, error_output_buffer_.allocation);
    }

    ClearCmdErrorsCountsBuffer();
    if (gpuav->aborted_) return;

    // If instrumentation found an error, skip post processing. Errors detected by instrumentation are usually
    // very serious, such as a prematurely destroyed resource and the state needed below is likely invalid.
    bool gpuav_success = false;
    if (!skip) {
        gpuav_success = ValidateBindlessDescriptorSets();
    }

    if (gpuav_success) {
        UpdateCmdBufImageLayouts(state_, *this);
    }
}

Queue::Queue(Validator &state, VkQueue q, uint32_t index, VkDeviceQueueCreateFlags flags, const VkQueueFamilyProperties &qfp)
    : gpu_tracker::Queue(state, q, index, flags, qfp) {}

vvl::PreSubmitResult Queue::PreSubmit(std::vector<vvl::QueueSubmission> &&submissions) {
    return gpu_tracker::Queue::PreSubmit(std::move(submissions));
}

void RestorablePipelineState::Create(vvl::CommandBuffer &cb_state, VkPipelineBindPoint bind_point) {
    cmd_buffer_ = cb_state.VkHandle();
    pipeline_bind_point_ = bind_point;
    const auto lv_bind_point = ConvertToLvlBindPoint(bind_point);

    LastBound &last_bound = cb_state.lastBound[lv_bind_point];
    if (last_bound.pipeline_state) {
        pipeline_ = last_bound.pipeline_state->VkHandle();

    } else {
        assert(shader_objects_.empty());
        if (lv_bind_point == BindPoint_Graphics) {
            shader_objects_ = last_bound.GetAllBoundGraphicsShaders();
        } else if (lv_bind_point == BindPoint_Compute) {
            auto compute_shader = last_bound.GetShaderState(ShaderObjectStage::COMPUTE);
            if (compute_shader) {
                shader_objects_.emplace_back(compute_shader);
            }
        }
    }

    desc_set_pipeline_layout_ = last_bound.desc_set_pipeline_layout;

    push_constants_data_ = cb_state.push_constant_data_chunks;

    descriptor_sets_.reserve(last_bound.per_set.size());
    for (std::size_t i = 0; i < last_bound.per_set.size(); i++) {
        const auto &bound_descriptor_set = last_bound.per_set[i].bound_descriptor_set;
        if (bound_descriptor_set) {
            descriptor_sets_.push_back(std::make_pair(bound_descriptor_set->VkHandle(), static_cast<uint32_t>(i)));
            if (bound_descriptor_set->IsPushDescriptor()) {
                push_descriptor_set_index_ = static_cast<uint32_t>(i);
            }
            dynamic_offsets_.push_back(last_bound.per_set[i].dynamicOffsets);
        }
    }

    if (last_bound.push_descriptor_set) {
        push_descriptor_set_writes_ = last_bound.push_descriptor_set->GetWrites();
    }
}

void RestorablePipelineState::Restore() const {
    if (pipeline_ != VK_NULL_HANDLE) {
        DispatchCmdBindPipeline(cmd_buffer_, pipeline_bind_point_, pipeline_);
    }
    if (!shader_objects_.empty()) {
        std::vector<VkShaderStageFlagBits> stages;
        std::vector<VkShaderEXT> shaders;
        for (const vvl::ShaderObject *shader_obj : shader_objects_) {
            stages.emplace_back(shader_obj->create_info.stage);
            shaders.emplace_back(shader_obj->VkHandle());
        }
        DispatchCmdBindShadersEXT(cmd_buffer_, static_cast<uint32_t>(shader_objects_.size()), stages.data(), shaders.data());
    }

    for (std::size_t i = 0; i < descriptor_sets_.size(); i++) {
        VkDescriptorSet descriptor_set = descriptor_sets_[i].first;
        if (descriptor_set != VK_NULL_HANDLE) {
            DispatchCmdBindDescriptorSets(cmd_buffer_, pipeline_bind_point_, desc_set_pipeline_layout_, descriptor_sets_[i].second,
                                          1, &descriptor_set, static_cast<uint32_t>(dynamic_offsets_[i].size()),
                                          dynamic_offsets_[i].data());
        }
    }

    if (!push_descriptor_set_writes_.empty()) {
        DispatchCmdPushDescriptorSetKHR(cmd_buffer_, pipeline_bind_point_, desc_set_pipeline_layout_, push_descriptor_set_index_,
                                        static_cast<uint32_t>(push_descriptor_set_writes_.size()),
                                        reinterpret_cast<const VkWriteDescriptorSet *>(push_descriptor_set_writes_.data()));
    }

    for (const auto &push_constant_range : push_constants_data_) {
        DispatchCmdPushConstants(cmd_buffer_, push_constant_range.layout, push_constant_range.stage_flags,
                                 push_constant_range.offset, static_cast<uint32_t>(push_constant_range.values.size()),
                                 push_constant_range.values.data());
    }
}

void ValidationPipeline::Destroy() {
    device_ = VK_NULL_HANDLE;

    if (pipeline_layout_ != VK_NULL_HANDLE) {
        DispatchDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (shader_object_ != VK_NULL_HANDLE) {
        DispatchDestroyShaderEXT(device_, shader_object_, nullptr);
        shader_object_ = VK_NULL_HANDLE;
    }

    if (shader_module_ != VK_NULL_HANDLE) {
        DispatchDestroyShaderModule(device_, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
    }

    if (pipeline_ != VK_NULL_HANDLE) {
        DispatchDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
}

void ValidationPipeline::SetDescriptorSetLayouts(uint32_t set_layout_count, const VkDescriptorSetLayout *set_layouts) {
    pipeline_layout_ci_.setLayoutCount = set_layout_count;
    pipeline_layout_ci_.pSetLayouts = set_layouts;
}

void ValidationPipeline::SetPushConstantRanges(uint32_t ranges_count, const VkPushConstantRange *ranges) {
    pipeline_layout_ci_.pushConstantRangeCount = ranges_count;
    pipeline_layout_ci_.pPushConstantRanges = ranges;
}

void ValidationPipeline::SetComputeShader(bool uses_shader_object, size_t code_dwords_count, const uint32_t *code) {
    SetShader(uses_shader_object, VK_SHADER_STAGE_COMPUTE_BIT, code_dwords_count, code);
}

void ValidationPipeline::SetVertexShader(bool uses_shader_object, size_t code_dwords_count, const uint32_t *code) {
    SetShader(uses_shader_object, VK_SHADER_STAGE_VERTEX_BIT, code_dwords_count, code);
}

bool ValidationPipeline::BuildOnlyLayoutAndShader(Validator &gpuav, const Location &loc) {
    assert(device_ == VK_NULL_HANDLE);
    device_ = gpuav.device;
    assert(pipeline_layout_ == VK_NULL_HANDLE);
    VkResult result = DispatchCreatePipelineLayout(gpuav.device, &pipeline_layout_ci_, nullptr, &pipeline_layout_);
    if (result != VK_SUCCESS) {
        gpuav.InternalError(gpuav.device, loc,
                            "Unable to create pipeline layout for SharedDrawValidationResources. Aborting GPU-AV.");
        return false;
    }

    const bool uses_shader_object = shader_object_ci_.codeSize != 0;
    if (uses_shader_object) {
        shader_object_ci_.setLayoutCount = pipeline_layout_ci_.setLayoutCount;
        shader_object_ci_.pSetLayouts = pipeline_layout_ci_.pSetLayouts;
        shader_object_ci_.pushConstantRangeCount = pipeline_layout_ci_.pushConstantRangeCount;
        shader_object_ci_.pPushConstantRanges = pipeline_layout_ci_.pPushConstantRanges;
        result = DispatchCreateShadersEXT(gpuav.device, 1u, &shader_object_ci_, nullptr, &shader_object_);
        if (result != VK_SUCCESS) {
            gpuav.InternalError(gpuav.device, loc, "Unable to create shader object. Aborting GPU-AV.");
            return false;
        }
    } else {
        result = DispatchCreateShaderModule(gpuav.device, &shader_module_ci_, nullptr, &shader_module_);
        if (result != VK_SUCCESS) {
            gpuav.InternalError(gpuav.device, loc, "Unable to create shader module. Aborting GPU-AV.");
            return false;
        }

        VkPipelineShaderStageCreateInfo pipeline_stage_ci = vku::InitStructHelper();
        pipeline_stage_ci.stage = shader_stage_;
        pipeline_stage_ci.module = shader_module_;
        pipeline_stage_ci.pName = "main";

        VkComputePipelineCreateInfo pipeline_ci = vku::InitStructHelper();
        pipeline_ci.stage = pipeline_stage_ci;
        pipeline_ci.layout = pipeline_layout_;
    }

    return true;
}

bool ValidationPipeline::BuildPipeline(Validator &gpuav, const Location &loc) {
    const bool uses_shader_object = shader_object_ci_.codeSize != 0;

    if (uses_shader_object) {
        return true;
    }

    VkResult result = DispatchCreateShaderModule(gpuav.device, &shader_module_ci_, nullptr, &shader_module_);
    if (result != VK_SUCCESS) {
        gpuav.InternalError(gpuav.device, loc, "Unable to create shader module. Aborting GPU-AV.");
        return false;
    }

    if (shader_stage_ == VK_SHADER_STAGE_COMPUTE_BIT) {
        VkPipelineShaderStageCreateInfo pipeline_stage_ci = vku::InitStructHelper();
        pipeline_stage_ci.stage = shader_stage_;
        pipeline_stage_ci.module = shader_module_;
        pipeline_stage_ci.pName = "main";

        VkComputePipelineCreateInfo pipeline_ci = vku::InitStructHelper();
        pipeline_ci.stage = pipeline_stage_ci;
        pipeline_ci.layout = pipeline_layout_;
        pipeline_ci.stage = pipeline_stage_ci;

        result = DispatchCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline_);
    } else if (shader_stage_ == VK_SHADER_STAGE_VERTEX_BIT) {
        // #ARNO_TODO ValidationPipeline::BuildPipeline for graphics pipeline
        assert(false);
    }

    DispatchDestroyShaderModule(gpuav.device, shader_module_, nullptr);
    shader_module_ = VK_NULL_HANDLE;

    if (result != VK_SUCCESS) {
        gpuav.InternalError(gpuav.device, loc, "Failed to create pipeline. Aborting GPU-AV.");
        return false;
    }

    return true;
}

void ValidationPipeline::Bind(VkCommandBuffer cmd_buffer) const {
    const bool uses_shader_object = shader_object_ci_.codeSize != 0;

    if (uses_shader_object) {
        VkShaderStageFlagBits stage = shader_stage_;
        DispatchCmdBindShadersEXT(cmd_buffer, 1u, &stage, &shader_object_);
    } else {
        VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_MAX_ENUM;
        if (shader_stage_ == VK_SHADER_STAGE_COMPUTE_BIT) {
            bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        } else if (shader_stage_ == VK_SHADER_STAGE_VERTEX_BIT) {
            bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
        } else if (shader_stage_ == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
            bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
        }

        DispatchCmdBindPipeline(cmd_buffer, bind_point, pipeline_);
    }
}

void ValidationPipeline::SetShader(bool uses_shader_object, VkShaderStageFlagBits shader_stage, size_t code_dwords_count,
                                   const uint32_t *code) {
    assert(device_ == VK_NULL_HANDLE && "limitation: can only set one shader, and once");
    shader_stage_ = shader_stage;

    if (uses_shader_object) {
        shader_object_ci_.stage = shader_stage;
        shader_object_ci_.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
        shader_object_ci_.codeSize = code_dwords_count * sizeof(uint32_t);
        shader_object_ci_.pCode = code;
        shader_object_ci_.pName = "main";
    } else {
        shader_module_ci_.codeSize = code_dwords_count * sizeof(uint32_t);
        shader_module_ci_.pCode = code;
    }
}

}  // namespace gpuav