/*
 * Copyright (c) 2015-2025 The Khronos Group Inc.
 * Copyright (c) 2015-2025 Valve Corporation
 * Copyright (c) 2015-2025 LunarG, Inc.
 * Copyright (c) 2015-2025 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/sync_helper.h"

class PositiveBuffer : public VkLayerTest {};

TEST_F(PositiveBuffer, OwnershipTranfers) {
    TEST_DESCRIPTION("Valid buffer ownership transfers that shouldn't create errors");
    RETURN_IF_SKIP(Init());

    vkt::Queue *no_gfx_queue = m_device->QueueWithoutCapabilities(VK_QUEUE_GRAPHICS_BIT);
    if (!no_gfx_queue) {
        GTEST_SKIP() << "Required queue not present (non-graphics non-compute capable required)";
    }

    vkt::CommandPool no_gfx_pool(*m_device, no_gfx_queue->family_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    vkt::CommandBuffer no_gfx_cb(*m_device, no_gfx_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
    auto buffer_barrier = buffer.BufferMemoryBarrier(0, 0, 0, VK_WHOLE_SIZE);

    // Let gfx own it.
    buffer_barrier.srcQueueFamilyIndex = m_device->graphics_queue_node_index_;
    buffer_barrier.dstQueueFamilyIndex = m_device->graphics_queue_node_index_;
    ValidOwnershipTransferOp(m_errorMonitor, m_default_queue, m_command_buffer, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, &buffer_barrier, nullptr);

    // Transfer it to non-gfx
    buffer_barrier.dstQueueFamilyIndex = no_gfx_queue->family_index;
    ValidOwnershipTransfer(m_errorMonitor, m_default_queue, m_command_buffer, no_gfx_queue, no_gfx_cb,
                           VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, &buffer_barrier, nullptr);

    // Transfer it to gfx
    buffer_barrier.srcQueueFamilyIndex = no_gfx_queue->family_index;
    buffer_barrier.dstQueueFamilyIndex = m_device->graphics_queue_node_index_;
    ValidOwnershipTransfer(m_errorMonitor, no_gfx_queue, no_gfx_cb, m_default_queue, m_command_buffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, &buffer_barrier, nullptr);
}

TEST_F(PositiveBuffer, TexelBufferAlignmentIn13) {
    TEST_DESCRIPTION("texelBufferAlignment is enabled by default in 1.3.");

    SetTargetApiVersion(VK_API_VERSION_1_3);
    RETURN_IF_SKIP(Init());

    const VkDeviceSize minTexelBufferOffsetAlignment = m_device->Physical().limits_.minTexelBufferOffsetAlignment;
    if (minTexelBufferOffsetAlignment == 1) {
        GTEST_SKIP() << "Test requires minTexelOffsetAlignment to not be equal to 1";
    }
    if (!BufferFormatAndFeaturesSupported(Gpu(), VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)) {
        GTEST_SKIP() << "Test requires support for VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT";
    }

    VkPhysicalDeviceVulkan13Properties props_1_3 = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(props_1_3);
    if (props_1_3.uniformTexelBufferOffsetAlignmentBytes < 4 || !props_1_3.uniformTexelBufferOffsetSingleTexelAlignment) {
        GTEST_SKIP() << "need uniformTexelBufferOffsetAlignmentBytes to be more than 4 with "
                        "uniformTexelBufferOffsetSingleTexelAlignment support";
    }

    // to prevent VUID-VkBufferViewCreateInfo-buffer-02751
    const uint32_t block_size = 4;  // VK_FORMAT_R8G8B8A8_UNORM
    vkt::Buffer buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);

    VkBufferViewCreateInfo buff_view_ci = vku::InitStructHelper();
    buff_view_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    buff_view_ci.range = VK_WHOLE_SIZE;
    buff_view_ci.buffer = buffer;
    buff_view_ci.offset = minTexelBufferOffsetAlignment + block_size;
    vkt::BufferView buffer_view(*m_device, buff_view_ci);
}

// The two PerfGetBufferAddress tests are intended to be used locally to monitor performance of the internal address -> buffer map
TEST_F(PositiveBuffer, DISABLED_PerfGetBufferAddressWorstCase) {
    TEST_DESCRIPTION("Add elements to buffer_address_map, worst case scenario");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(Init());

    // Allocate common buffer memory, all buffers will be bound to it so that they have the same starting address
    VkMemoryAllocateFlagsInfo alloc_flags = vku::InitStructHelper();
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo alloc_info = vku::InitStructHelper(&alloc_flags);
    alloc_info.allocationSize = 100 * 4096 * 4096;
    vkt::DeviceMemory buffer_memory(*m_device, alloc_info);

    // Create buffers. They have the same starting offset, but a growing size.
    // This is the worst case scenario for adding an element in the current buffer_address_map: inserted range will have to be split
    // for every range currently in the map.
    constexpr size_t N = 1400;
    std::vector<vkt::Buffer> buffers(N);
    VkBufferCreateInfo buffer_ci = vku::InitStructHelper();
    buffer_ci.size = 4096;
    buffer_ci.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkDeviceAddress ref_address = 0;

    for (size_t i = 0; i < N; ++i) {
        vkt::Buffer &buffer = buffers[i];
        buffer_ci.size = (i + 1) * 4096;
        buffer.InitNoMemory(*m_device, buffer_ci);
        vk::BindBufferMemory(device(), buffer, buffer_memory, 0);
        VkDeviceAddress addr = buffer.Address();
        if (ref_address == 0) {
            ref_address = addr;
        }
        if (addr != ref_address) {
            GTEST_SKIP() << "At iteration " << i << ", retrieved buffer address (" << addr << ") != reference address ("
                         << ref_address << ")";
        }
    }
}

// The two PerfGetBufferAddress tests are intended to be used locally to monitor performance of the internal address -> buffer map
TEST_F(PositiveBuffer, DISABLED_PerfGetBufferAddressGoodCase) {
    TEST_DESCRIPTION("Add elements to buffer_address_map, good case scenario");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(Init());

    // Allocate common buffer memory, all buffers will be bound to it so that they have the same starting address
    VkMemoryAllocateFlagsInfo alloc_flags = vku::InitStructHelper();
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo alloc_info = vku::InitStructHelper(&alloc_flags);
    alloc_info.allocationSize = 100 * 4096 * 4096;
    vkt::DeviceMemory buffer_memory(*m_device, alloc_info);

    // Create buffers. They have consecutive device address ranges, so no overlaps: no split will be needed when inserting, it
    // should be fast.
    constexpr size_t N = 1400;  // 100 * 4096;
    std::vector<vkt::Buffer> buffers(N);
    VkBufferCreateInfo buffer_ci = vku::InitStructHelper();
    buffer_ci.size = 4096;
    buffer_ci.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    for (size_t i = 0; i < N; ++i) {
        vkt::Buffer &buffer = buffers[i];
        buffer.InitNoMemory(*m_device, buffer_ci);
        // Consecutive offsets
        vk::BindBufferMemory(device(), buffer, buffer_memory, i * buffer_ci.size);
        (void)buffer.Address();
    }
}

TEST_F(PositiveBuffer, IndexBuffer2Size) {
    TEST_DESCRIPTION("Valid vkCmdBindIndexBuffer2KHR");
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance5);
    RETURN_IF_SKIP(Init());
    InitRenderTarget();

    const uint32_t buffer_size = 32;
    vkt::Buffer buffer(*m_device, buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindIndexBuffer2KHR(m_command_buffer, buffer, 4, 8, VK_INDEX_TYPE_UINT32);

    vk::CmdBindIndexBuffer2KHR(m_command_buffer, buffer, 0, buffer_size, VK_INDEX_TYPE_UINT32);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
}

TEST_F(PositiveBuffer, IndexBufferNull) {
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_6_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance6);
    RETURN_IF_SKIP(Init());
    InitRenderTarget();

    CreatePipelineHelper pipe(*this);
    pipe.CreateGraphicsPipeline();

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vk::CmdBindIndexBuffer(m_command_buffer, VK_NULL_HANDLE, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdDrawIndexed(m_command_buffer, 0, 1, 0, 0, 0);
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
}

TEST_F(PositiveBuffer, BufferViewUsageBasic) {
    TEST_DESCRIPTION("VkBufferUsageFlags2CreateInfoKHR with good flags.");
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance5);
    RETURN_IF_SKIP(Init());

    vkt::Buffer buffer(*m_device, 32, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);

    VkBufferUsageFlags2CreateInfoKHR buffer_usage_flags = vku::InitStructHelper();
    buffer_usage_flags.usage = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT;

    VkBufferViewCreateInfo buffer_view_ci = vku::InitStructHelper(&buffer_usage_flags);
    buffer_view_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    buffer_view_ci.range = VK_WHOLE_SIZE;
    buffer_view_ci.buffer = buffer;
    vkt::BufferView buffer_view(*m_device, buffer_view_ci);
}

TEST_F(PositiveBuffer, BufferUsageFlags2Subset) {
    TEST_DESCRIPTION("VkBufferUsageFlags2CreateInfoKHR that are a subset of the Buffer.");
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance5);
    RETURN_IF_SKIP(Init());

    vkt::Buffer buffer(*m_device, 32, VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT);

    VkBufferUsageFlags2CreateInfoKHR buffer_usage_flags = vku::InitStructHelper();
    buffer_usage_flags.usage = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT;

    VkBufferViewCreateInfo buffer_view_ci = vku::InitStructHelper(&buffer_usage_flags);
    buffer_view_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    buffer_view_ci.range = VK_WHOLE_SIZE;
    buffer_view_ci.buffer = buffer;
    vkt::BufferView buffer_view(*m_device, buffer_view_ci);
}

TEST_F(PositiveBuffer, BufferUsageFlags2Ignore) {
    TEST_DESCRIPTION("Ignore old flags if using VkBufferUsageFlags2CreateInfoKHR.");
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance5);
    RETURN_IF_SKIP(Init());

    VkBufferUsageFlags2CreateInfoKHR buffer_usage_flags = vku::InitStructHelper();
    buffer_usage_flags.usage = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT;

    VkBufferCreateInfo buffer_ci = vku::InitStructHelper(&buffer_usage_flags);
    buffer_ci.size = 32;
    buffer_ci.usage = VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT;
    vkt::Buffer buffer(*m_device, buffer_ci, vkt::no_mem);

    buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_VIDEO_DECODE_DST_BIT_KHR |
                      VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT;
    vkt::Buffer buffer2(*m_device, buffer_ci, vkt::no_mem);
}

TEST_F(PositiveBuffer, BufferUsageFlags2Usage) {
    TEST_DESCRIPTION("Ignore old flags if using VkBufferUsageFlags2CreateInfoKHR, even if bad.");
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::maintenance5);
    RETURN_IF_SKIP(Init());

    VkBufferUsageFlags2CreateInfoKHR buffer_usage_flags = vku::InitStructHelper();
    buffer_usage_flags.usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;

    VkBufferCreateInfo buffer_ci = vku::InitStructHelper(&buffer_usage_flags);
    buffer_ci.size = 32;
    buffer_ci.usage = 0;
    vkt::Buffer buffer(*m_device, buffer_ci, vkt::no_mem);

    buffer_ci.usage = 0xBAD0000;
    vkt::Buffer buffer2(*m_device, buffer_ci, vkt::no_mem);
}

TEST_F(PositiveBuffer, ReadBeforePointerPushConstant) {
    TEST_DESCRIPTION("Read before the valid pointer - use Push Constants to set the value");
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);

    RETURN_IF_SKIP(Init());
    InitRenderTarget();

    OneOffDescriptorSet descriptor_set(m_device, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr}});

    VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT, 0, 2 * sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci = vku::InitStructHelper();
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descriptor_set.layout_.handle();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push_constant_ranges;

    vkt::PipelineLayout pipeline_layout(*m_device, plci);

    char const *shader_source = R"glsl(
            #version 450
            #extension GL_EXT_buffer_reference : enable
            #extension GL_ARB_gpu_shader_int64 : require

            layout(buffer_reference, buffer_reference_align = 16) buffer ErrorsBuffer {
                uint size;
                uint written_count;
                uint data[];
            };
            
            layout(buffer_reference, buffer_reference_align = 16) buffer ActionIndexBuffer { 
                uint index[]; 
            };
            
            layout(buffer_reference, buffer_reference_align = 16) buffer CmdResourceIndexBuffer { 
                uint index[]; 
            };
            
            layout(buffer_reference, buffer_reference_align = 16) buffer CmdErrorsCountBuffer { 
                uint errors_count[]; 
            };    

            // Represent a [begin, end) range, where end is one past the last element held in range
            struct Range {
                uint64_t begin;
                uint64_t end;
            };
            
            layout(buffer_reference) buffer BDARanges {
                uint ranges_count;
                uint padding_unused;
                Range ranges[];
            };
            
            // Ranges are supposed to:
            // 1) be stored from low to high
            // 2) not overlap
            layout(buffer_reference) buffer BDAInputBuffer {
                BDARanges bda_ranges_ptr;
            };

            layout(buffer_reference, buffer_reference_align = 16) buffer RootNode { 
                ErrorsBuffer inst_errors_buffer;
                ActionIndexBuffer inst_action_index_buffer;
                CmdResourceIndexBuffer inst_cmd_resource_index_buffer;
                CmdErrorsCountBuffer inst_cmd_errors_count_buffer;

                BDAInputBuffer bda_input_buffer;
            };

            layout(set = 0, binding = 0, std430) buffer RootNodeBuffer { 
              RootNode root_node; 
            };
            
            // It is common that an app only has a single BDA address and it is used to poke inside a struct.
            // This likely means there is only a single range being accessed, and for shader that do multiple checks,
            // we can hopefully speed up runtime perf by hitting this early and leaving fast
            //
            // TODO - Play around more with having the cache be the last 2 or 4 elements as well as having no cache
            //        (and picking depending on what we see instrumenting)
            //
            // Note - This NEEDS to be initialized with zero otherwise found to crash drivers
            //        (it will print as zero, but if used to index into an array, will just crash).
            //        GLSL lacks the ability to use the Initializer ID to a OpVariable, so while linking,
            //        we will adjust the SPIR-V to set this to zero to start
            uint index_cache;
            
            bool inst_buffer_device_address_range(
                const uint inst_num,
                const uint64_t addr,
                const uint access_type,
                const uint access_byte_size)
            {
                //const Range cache_range = root_node.bda_input_buffer.bda_ranges_ptr.ranges[index_cache];
                //if (addr >= cache_range.begin && ((addr + access_byte_size) <= cache_range.end)) {
                //    return true;
                //}

                // Find out if addr is valid
                // ---
                for (uint range_i = 0; range_i < root_node.bda_input_buffer.bda_ranges_ptr.ranges_count; ++range_i) {
                    Range range = root_node.bda_input_buffer.bda_ranges_ptr.ranges[range_i];
                    if (addr < range.begin) {
                        // Invalid address, proceed to error logging
                        break;
                    }
                    if ((addr < range.end) && (addr + access_byte_size > range.end)) {
                        // Ranges do not overlap,
                        // so if current range holds addr but not (add + access_byte_size), access is invalid
                        break;
                    }
                    if ((addr + access_byte_size) <= range.end) {
                        // addr >= range.begin && addr + access_byte_size <= range.end
                        // ==> valid access
                        // index_cache = range_i;
                        return true;
                    }
                    // Address is above current range, proceed to next range.
                    // If at loop end, address is invalid.
                }
                return false;
            }

            layout(buffer_reference, std430, buffer_reference_align = 16) buffer bufStruct {
                uint a[4];
            };

            layout(push_constant) uniform ufoo {
                bufStruct data;
                uint nWrites;
            } u_info;
            
            void LogError(const uint payload_0, const uint payload_1) {
                const uint cmd_id = root_node.inst_cmd_resource_index_buffer.index[0];
                const uint cmd_errors_count = atomicAdd(root_node.inst_cmd_errors_count_buffer.errors_count[cmd_id], 1);
                const uint kMaxErrorsPerCmd = 6;
                const bool max_cmd_errors_count_reached = cmd_errors_count >= kMaxErrorsPerCmd;
                if (!max_cmd_errors_count_reached) {        
                    const uint kErrorRecordSize = 16;
                    uint write_pos = atomicAdd(root_node.inst_errors_buffer.written_count, kErrorRecordSize);
                    const bool errors_buffer_not_filled = (write_pos + kErrorRecordSize) <= uint(root_node.inst_errors_buffer.size);
                    if (errors_buffer_not_filled) {
                        root_node.inst_errors_buffer.data[write_pos + 0] = 16;
                        root_node.inst_errors_buffer.data[write_pos + 1] = 44;
                        root_node.inst_errors_buffer.data[write_pos + 2] = 71;
                        root_node.inst_errors_buffer.data[write_pos + 3] = 0;
                        root_node.inst_errors_buffer.data[write_pos + 4] = 1;
                        root_node.inst_errors_buffer.data[write_pos + 5] = 2;
                        root_node.inst_errors_buffer.data[write_pos + 6] = kMaxErrorsPerCmd;
                        root_node.inst_errors_buffer.data[write_pos + 7] = cmd_id;
                        root_node.inst_errors_buffer.data[write_pos + 8] = cmd_errors_count;
                        root_node.inst_errors_buffer.data[write_pos + 9] = payload_0;                    
                        root_node.inst_errors_buffer.data[write_pos + 10] = payload_1;                    
                    }
                }
            }

            void main() {
                for (uint i=0; i < u_info.nWrites; ++i) {
                    if (!inst_buffer_device_address_range(17, uint64_t(u_info.data) + 4 * i, 1, 4)) 
                    {
                        LogError( uint(uint64_t(u_info.data) + 4 * i), uint((uint64_t(u_info.data) + 4 * i) >> uint(32)) );
                        // LogError( uint(uint64_t(root_node.bda_input_buffer)), uint(uint64_t(root_node.bda_input_buffer.bda_ranges_ptr)) );
                        continue;
                    }

                    u_info.data.a[i] = 66;
                   
                }
            }
        )glsl";

    VkShaderObj vs(this, shader_source, VK_SHADER_STAGE_VERTEX_BIT);

    CreatePipelineHelper pipe(*this);
    pipe.shader_stages_[0] = vs.GetStageCreateInfo();
    pipe.gp_ci_.layout = pipeline_layout;
    pipe.CreateGraphicsPipeline();

    // Setup buffers mimicking GPU-AV error logging
    vkt::Buffer errors_buffer(*m_device, 64 * sizeof(uint32_t), 0, vkt::device_address);
    auto errors_buffer_ptr = (uint32_t *)errors_buffer.Memory().Map();
    std::memset(errors_buffer_ptr, 0, 64 * sizeof(uint32_t));
    errors_buffer_ptr[0] = 64;

    vkt::Buffer indices_buffer(*m_device, 64 * sizeof(uint32_t), 0, vkt::device_address);
    auto indices_buffer_ptr = (uint32_t *)indices_buffer.Memory().Map();
    std::memset(indices_buffer_ptr, 0, 64 * sizeof(uint32_t));
    const uint32_t indices_buffer_alignment_ = 8 * sizeof(uint32_t);
    for (uint32_t i = 0; i < indices_buffer.CreateInfo().size / sizeof(uint32_t); ++i) {
        indices_buffer_ptr[i] = i / (indices_buffer_alignment_ / sizeof(uint32_t));
    }

    vkt::Buffer errors_count_buffer(*m_device, 64 * sizeof(uint32_t), 0, vkt::device_address);
    auto errors_count_buffer_ptr = (uint32_t *)errors_count_buffer.Memory().Map();
    std::memset(errors_count_buffer_ptr, 0, 64 * sizeof(uint32_t));

    vkt::Buffer root_node_buffer(*m_device, 8 * sizeof(VkDeviceAddress), 0, vkt::device_address);
    auto root_node_buffer_ptr = (VkDeviceAddress *)root_node_buffer.Memory().Map();
    std::memset(root_node_buffer_ptr, 0, 64 * sizeof(uint32_t));
    root_node_buffer_ptr[0] = errors_buffer.Address();
    root_node_buffer_ptr[1] = indices_buffer.Address();
    root_node_buffer_ptr[2] = indices_buffer.Address();
    root_node_buffer_ptr[3] = errors_count_buffer.Address();

    vkt::Buffer root_node_addr_buffer(*m_device, sizeof(VkDeviceAddress), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);
    auto root_node_addr_buffer_ptr = (VkDeviceAddress *)root_node_addr_buffer.Memory().Map();
    *root_node_addr_buffer_ptr = root_node_buffer.Address();

    descriptor_set.WriteDescriptorBufferInfo(0, root_node_addr_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    descriptor_set.UpdateDescriptorSets();

    vkt::Buffer uniform_buffer(*m_device, 4 * sizeof(uint32_t), 0, vkt::device_address);

    vkt::Buffer bda_ranges_buffer(*m_device, 2 * sizeof(uint32_t) + 1 * (2 * sizeof(uint64_t)), 0, vkt::device_address);
    auto bda_ranges_buffer_ptr = bda_ranges_buffer.Memory().Map();
    auto bda_ranges_buffer_u32_ptr = (uint32_t *)bda_ranges_buffer_ptr;
    auto bda_ranges_buffer_u64_ptr = (uint64_t *)bda_ranges_buffer_ptr;
    bda_ranges_buffer_u32_ptr[0] = 1;
    bda_ranges_buffer_u32_ptr[1] = 0;
    bda_ranges_buffer_u64_ptr[1] = uniform_buffer.Address();
    bda_ranges_buffer_u64_ptr[2] = uniform_buffer.Address() + uniform_buffer.CreateInfo().size;

    vkt::Buffer bda_ranges_buffer_input_buffer(*m_device, sizeof(VkDeviceAddress), 0, vkt::device_address);
    auto bda_ranges_buffer_input_buffer_ptr = (VkDeviceAddress *)bda_ranges_buffer_input_buffer.Memory().Map();
    *bda_ranges_buffer_input_buffer_ptr = bda_ranges_buffer.Address();

    root_node_buffer_ptr[4] = bda_ranges_buffer_input_buffer.Address();

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    // Shader will try to write to 1 invalid address
    const VkDeviceAddress push_constants_addr = uniform_buffer.Address() - 4;
    vk::CmdPushConstants(m_command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VkDeviceAddress),
                         &push_constants_addr);
    const uint32_t push_constant_n_writes = 4;
    vk::CmdPushConstants(m_command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(VkDeviceAddress), sizeof(uint32_t),
                         &push_constant_n_writes);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set.set_, 0,
                              nullptr);

    vk::CmdDraw(m_command_buffer, 3, 1, 0, 0);
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();

    // Make sure we wrote the other 3 values
    auto *buffer_ptr = (uint32_t *)(uniform_buffer.Memory().Map());
    ASSERT_EQ(buffer_ptr[0], 66);
    ASSERT_EQ(buffer_ptr[1], 66);
    ASSERT_EQ(buffer_ptr[2], 66);

    // Make sure we wrote an error
    const uint32_t write_pos = 2;
    ASSERT_EQ(errors_buffer_ptr[write_pos + 0], 16);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 1], 44);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 2], 71);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 3], 0);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 4], 1);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 5], 2);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 6], 6);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 7], 0);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 8], 0);
    ASSERT_EQ(errors_buffer_ptr[write_pos + 9], uint32_t(uniform_buffer.Address() - 4));
    ASSERT_EQ(errors_buffer_ptr[write_pos + 10], uint32_t((uniform_buffer.Address() - 4) >> 32u));
}
