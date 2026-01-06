/***************************************************************************
 *
 * Copyright (c) 2015-2025 The Khronos Group Inc.
 * Copyright (c) 2015-2025 Valve Corporation
 * Copyright (c) 2015-2025 LunarG, Inc.
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
 ****************************************************************************/

// NOLINTBEGIN

#include <array>
#include <cstring>
#include <mutex>

#include "chassis/validation_object.h"

#include "generated/dispatch_functions.h"

namespace vvl::base {

void Instance::CommonPostCallRecordEnumeratePhysicalDevice(const VkPhysicalDevice *phys_devices, const int count) {
    // Assume phys_devices is valid
    assert(phys_devices);
    for (int i = 0; i < count; ++i) {
        const auto &phys_device = phys_devices[i];
        if (0 == physical_device_properties_map.count(phys_device)) {
            VkPhysicalDeviceProperties phys_dev_props = {};
            DispatchGetPhysicalDeviceProperties(phys_device, &phys_dev_props);
            physical_device_properties_map[phys_device] = phys_dev_props;

            // Enumerate the Device Ext Properties to save the PhysicalDevice supported extension state
            uint32_t ext_count = 0;

            std::vector<VkExtensionProperties> ext_props{};
            DispatchEnumerateDeviceExtensionProperties(phys_device, nullptr, &ext_count, nullptr);
            ext_props.resize(ext_count);
            DispatchEnumerateDeviceExtensionProperties(phys_device, nullptr, &ext_count, ext_props.data());

            DeviceExtensions phys_dev_exts(extensions, phys_dev_props.apiVersion, ext_props);
            printf("Went through Instance::CommonPostCallRecordEnumeratePhysicalDevice\n");
            printf("phys_dev_exts.vk_khr_maintenance5.ext_enabled: %d\n", phys_dev_exts.vk_khr_maintenance5.ext_enabled);
            physical_device_extensions[phys_device] = std::move(phys_dev_exts);  // #ARNO_TODO ah, everyone needs to use that
        }
    }
}

void Instance::PostCallRecordEnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                      VkPhysicalDevice *pPhysicalDevices, const RecordObject &record_obj) {
    if ((VK_SUCCESS != record_obj.result) && (VK_INCOMPLETE != record_obj.result)) {
        return;
    }

    if (!pPhysicalDeviceCount || !pPhysicalDevices) {
        return;
    }
    CommonPostCallRecordEnumeratePhysicalDevice(pPhysicalDevices, *pPhysicalDeviceCount);
}

void Instance::PostCallRecordEnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t *pPhysicalDeviceGroupCount,
                                                           VkPhysicalDeviceGroupProperties *pPhysicalDeviceGroupProperties,
                                                           const RecordObject &record_obj) {
    if ((VK_SUCCESS != record_obj.result) && (VK_INCOMPLETE != record_obj.result)) {
        return;
    }

    if (pPhysicalDeviceGroupCount && pPhysicalDeviceGroupProperties) {
        for (uint32_t i = 0; i < *pPhysicalDeviceGroupCount; i++) {
            const auto &group = pPhysicalDeviceGroupProperties[i];
            CommonPostCallRecordEnumeratePhysicalDevice(group.physicalDevices, group.physicalDeviceCount);
        }
    }
}

}  // namespace vvl::base
