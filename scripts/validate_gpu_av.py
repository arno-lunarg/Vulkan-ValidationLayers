#!/usr/bin/env python3
#
# Copyright (c) 2016-2023 Valve Corporation
# Copyright (c) 2016-2023 LunarG, Inc.
# Copyright (c) 2016-2022 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
import struct
import re
import argparse

def set_env_var(name, value):
    '''Set an environment variable to a given value or remove it from the
    environment if None
    '''
    if value is not None:
        os.environ[name] = value
    elif name in os.environ:
        del os.environ[name]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Helper script to validate the Vulkan calls made by the validation layer to perform its job.
        First a trace of  program is captured with GFXReconstruct. Then this trace is replayed, with validation enabled.''')
    parser.add_argument('--gfxr-dir', type=str, required=True, help='Absolute path to GFXReconstruct binary folder')
    parser.add_argument('--gfxr-out', type=str, required=True, help='Absolute path to GFXReconstruct capture file')
    parser.add_argument('--vvl', type=str, required=True, help='Absolute path to the validation layer directory')
    parser.add_argument('-w', type=str, help='Program working directory')
    parser.add_argument(
        'program_and_args', metavar='<program> [<program args>]', nargs=argparse.REMAINDER,
        help='''Program to capture, optionally followed by program.
        If you are using this script to debug a gpu-av test, do not forget to add the --gpuav-enable-core option to the test .exe command line.''')
    args = parser.parse_args()

    # Set validation layer path
    set_env_var('VK_LAYER_PATH', args.vvl)
    
    # Capture GFXR trace
    gfxr_capture_script = os.path.join(args.gfxr_dir, "gfxrecon-capture-vulkan.py")
    gfxr_capture_script_args = ["python", gfxr_capture_script,  "--no-file-timestamp", "--log-level", "debug", "-o", args.gfxr_out]
    if args.w:
        gfxr_capture_script_args += ["-w", args.w]
    gfxr_capture_script_args += ["--file-flush"]
    gfxr_capture_script_args += args.program_and_args
    print('GFXReconstruct capture command:')
    print(' '.join(x for x in gfxr_capture_script_args))
    result = subprocess.run(gfxr_capture_script_args, capture_output=True)
    if 0 != result.returncode:
        print('GFXReconstruct capture errors:\n', result.stderr.decode('utf-8'))
    print('GFXReconstruct capture output:\n', result.stdout.decode('utf-8'))

    # Replay capture with validation
    gfxr_replay = os.path.join(args.gfxr_dir, "gfxrecon-replay")
    gfxr_replay_args = [gfxr_replay, "--validate", "--log-level", "debug", args.gfxr_out]
    result = subprocess.run(gfxr_replay_args, capture_output=True)
    if 0 != result.returncode:
        print('GFXReconstruct  errors:\n', result.stderr.decode('utf-8'))
    print('Output:\n', result.stdout.decode('utf-8')) 
