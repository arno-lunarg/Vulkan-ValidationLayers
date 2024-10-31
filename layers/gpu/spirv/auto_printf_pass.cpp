/* Copyright (c) 2024 LunarG, Inc.
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

#include "auto_printf_pass.h"
#include "module.h"
#include "gpu/shaders/gpuav_error_header.h"
#include "utils/vk_layer_utils.h"
#include <spirv/unified1/NonSemanticDebugPrintf.h>
#include <cstring>
#include <iostream>

namespace gpuav {
namespace spirv {

// Used between injections of a function
void AutoPrintfPass::Reset() { ext_import_id_ = 0; }

bool AutoPrintfPass::Run() {
#if 0
    std::cout << "At top auto printf Run:" << std::endl;
    for (const auto& inst : module_.annotations_) {
        std::cout << inst->DebugString() << std::endl;
    }
#endif

    // Look for OpExtInstImport NonSemantic.DebugPrintf
    // Add it if not there
    // ---
    bool has_debug_printf_instruction_import = false;
#if 1
    for (const auto& inst : module_.ext_inst_imports_) {
        const char* import_string = inst->GetAsString(2);
        if (strcmp(import_string, "NonSemantic.DebugPrintf") == 0) {
            ext_import_id_ = inst->Word(1);
            has_debug_printf_instruction_import = true;
            break;
        }
    }
#endif

    auto get_words_from_string = [](std::string_view str, std::vector<uint32_t>* dwords) {
        uint32_t dword = 0;
        for (size_t i = 0; i < str.length(); ++i) {
            auto c = (uint32_t)str[i];
            const uint32_t shifted_char_byte = c << 8 * (i % 4);
            dword |= shifted_char_byte;
            if ((i % 4) == 3) {
                dwords->push_back(dword);
                dword = 0;
            }
        }
        dword |= '\0' << (str.length() % 4);
        dwords->push_back(dword);
    };

    if (!has_debug_printf_instruction_import) {
        const uint32_t printf_import_id = module_.TakeNextId();
        std::vector<uint32_t> printf_import_dwords = {printf_import_id};
        const std::string_view non_semantic_debug_printf_str = "NonSemantic.DebugPrintf";

        get_words_from_string(non_semantic_debug_printf_str, &printf_import_dwords);
        std::unique_ptr<Instruction> printf_import =
            std::make_unique<Instruction>(1 + (uint32_t)printf_import_dwords.size(), spv::Op::OpExtInstImport);
        printf_import->Fill(printf_import_dwords);
        const char* written_str = printf_import->GetAsString(2);
        (void)written_str;
        assert(strcmp(written_str, non_semantic_debug_printf_str.data()) == 0);
        printf_import->SetResultTypeIndex();
        ext_import_id_ = printf_import->ResultId();
        module_.ext_inst_imports_.emplace_back(std::move(printf_import));
    }

    assert(ext_import_id_ != 0);

    // Go through entry points, add a printf of entry point name as first instruction
    for (const auto& entry_point_inst : module_.entry_points_) {
        const uint32_t entry_point_id = entry_point_inst->Word(2);
        for (const auto& function : module_.functions_) {
            const uint32_t function_id = function->GetDef().Word(2);
            if (entry_point_id != function_id) continue;

            uint32_t execution_mode = entry_point_inst->Word(1);
            std::cout << "Execution mode: " << execution_mode << std::endl;
            std::vector<uint32_t> ray_tracing_execution_modes = {5313, 5314, 5315, 5316, 5317, 5318};
            if (!IsValueIn(execution_mode, ray_tracing_execution_modes)) continue;

            // Found entry point function, create corresponding OpString:
            // %result_id = OpString "In <entry_point_name>"
            std::string entry_point_name = entry_point_inst->GetAsString(3);
#if 0
            if (entry_point_name == "RayGen" || entry_point_name == "ClosestHit0" || entry_point_name == "AnyHit1") {
                continue;
            }
#endif
            entry_point_name += '\n';
            std::cout << "Adding auto printf to " << entry_point_name << std::endl;
            // std::cout << "Entry point name: " << entry_point_name << std::endl;
            const uint32_t op_string_id = module_.TakeNextId();
            std::vector<uint32_t> op_string_words = {op_string_id};
            get_words_from_string(entry_point_name, &op_string_words);
            std::unique_ptr<Instruction> op_string_inst =
                std::make_unique<Instruction>(1 + (uint32_t)op_string_words.size(), spv::Op::OpString);

            op_string_inst->Fill(op_string_words);
            module_.debug_source_.emplace_back(std::move(op_string_inst));

            // Now add printf call

            std::unique_ptr<Instruction> print_inst = std::make_unique<Instruction>(6, spv::Op::OpExtInst);
            const Type& void_type = module_.type_manager_.GetTypeVoid();
            const uint32_t void_type_id = void_type.inst_.Word(1);
            const uint32_t print_id = module_.TakeNextId();
            std::vector<uint32_t> print_inst_dwords = {void_type_id, print_id, ext_import_id_, 1, op_string_id};
            print_inst->Fill(print_inst_dwords);
            std::unique_ptr<BasicBlock>& block_instructions = (*function->blocks_.begin());
            InstructionIt insert_pos = block_instructions->GetFirstInjectableInstrution();
            block_instructions->instructions_.insert(insert_pos, std::move(print_inst));

            ++instrumentations_count_;
        }
    }

    if (instrumentations_count_ == 0) {
        // assert(false);
        return false;
    }

    return true;
}

void AutoPrintfPass::PrintDebugInfo() { std::cout << "AutoPrintfPass instrumentation count: " << instrumentations_count_ << '\n'; }

}  // namespace spirv
}  // namespace gpuav
