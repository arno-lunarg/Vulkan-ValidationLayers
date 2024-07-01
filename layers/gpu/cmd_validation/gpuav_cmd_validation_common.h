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

#pragma once

#include "gpu/resources/gpuav_subclasses.h"

#include <utility>
#include <vector>

namespace gpuav {
class Validator;

void BindValidationCmdsCommonDescSet(const LockedSharedPtr<CommandBuffer, WriteLockGuard>& cmd_buffer_state,
                                     VkPipelineBindPoint bind_point, VkPipelineLayout pipeline_layout, uint32_t cmd_index,
                                     uint32_t error_logger_index);

VkDeviceAddress GetBufferDeviceAddress(Validator& gpuav, VkBuffer buffer, const Location& loc);

}  // namespace gpuav
