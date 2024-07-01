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

#include "gpu/cmd_validation/gpuav_cmd_validation_common.h"

#include "gpu/core/gpuav.h"
#include "gpu/core/gpuav_constants.h"
#include "gpu/resources/gpu_resources.h"
#include "gpu/shaders/gpu_shaders_constants.h"

namespace gpuav {

void BindValidationCmdsCommonDescSet(const LockedSharedPtr<CommandBuffer, WriteLockGuard> &cmd_buffer_state,
                                     VkPipelineBindPoint bind_point, VkPipelineLayout pipeline_layout, uint32_t cmd_index,
                                     uint32_t error_logger_index) {
    assert(cmd_index < cst::indices_count);
    assert(error_logger_index < cst::indices_count);
    std::array<uint32_t, 2> dynamic_offsets = {
        {cmd_index * static_cast<uint32_t>(sizeof(uint32_t)), error_logger_index * static_cast<uint32_t>(sizeof(uint32_t))}};
    DispatchCmdBindDescriptorSets(cmd_buffer_state->VkHandle(), bind_point, pipeline_layout, glsl::kDiagCommonDescriptorSet, 1,
                                  &cmd_buffer_state->GetValidationCmdCommonDescriptorSet(),
                                  static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data());
}

VkDeviceAddress GetBufferDeviceAddress(Validator &gpuav, VkBuffer buffer, const Location &loc) {
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8001
    // Setting enabled_features.bufferDeviceAddress to true in GpuShaderInstrumentor::PreCallRecordCreateDevice
    // when adding missing features will modify another validator object, one associated to VkInstance,
    // and "this" validator is associated to a device. enabled_features is not inherited, and besides
    // would be reset in GetEnabledDeviceFeatures.
    // The switch from the instance validator object to the device one happens in
    // `state_tracker.cpp`, `ValidationStateTracker::PostCallRecordCreateDevice`
    // TL;DR is the following type of sanity check is currently invalid, but it would be nice to have
    // assert(enabled_features.bufferDeviceAddress);

    VkBufferDeviceAddressInfo address_info = vku::InitStructHelper();
    address_info.buffer = buffer;
    if (gpuav.api_version >= VK_API_VERSION_1_2) {
        return DispatchGetBufferDeviceAddress(gpuav.device, &address_info);
    }
    if (IsExtEnabled(gpuav.device_extensions.vk_ext_buffer_device_address)) {
        return DispatchGetBufferDeviceAddressEXT(gpuav.device, &address_info);
    }
    if (IsExtEnabled(gpuav.device_extensions.vk_khr_buffer_device_address)) {
        return DispatchGetBufferDeviceAddressKHR(gpuav.device, &address_info);
    }
    return 0;
}

}  // namespace gpuav
