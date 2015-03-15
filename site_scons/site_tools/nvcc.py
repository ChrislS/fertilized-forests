"""SCons.Tool.nvcc

Tool-specific initialization for NVIDIA CUDA Compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
import os
import sys
import platform


def get_cuda_paths():
  """Determines CUDA {bin,lib,include} paths
  
  returns (bin_path,lib_path,inc_path)
  """

  # determine defaults
  is_64bits = sys.maxsize > 2**32
  if os.name == 'nt':
    if is_64bits:
        lib_add_dir = 'x64'
    else:
        lib_add_dir = 'Win32'
    bin_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin'
    lib_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\%s' % (lib_add_dir)
    inc_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include'
  elif os.name == 'posix':
    if is_64bits:
        lib_add_dir = 'lib64'
    else:
        lib_add_dir = 'lib'
    bin_path = '/usr/local/cuda/bin'
    lib_path = '/usr/local/cuda/%s' % (lib_add_dir)
    inc_path = '/usr/local/cuda/include'
  elif not 'CUDA_ROOT' in os.environ or \
       not 'CUDA_LIB_DIR' in os.environ or \
       not 'CUDA_INCLUDE_DIR' in os.environ:
    raise ValueError, 'Error: unknown OS.  Where is nvcc installed? ' +\
      'Edit me, or set CUDA_ROOT, CUDA_LIB_DIR and CUDA_INCLUDE_DIR as'+\
      'environment variables.'

  # override with environement variables
  if 'CUDA_ROOT' in os.environ:
    bin_path = os.path.join(os.path.abspath(os.environ['CUDA_ROOT']), 'bin')
  if 'CUDA_LIB_DIR' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_DIR'])
  if 'CUDA_INCLUDE_DIR' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INCLUDE_DIR'])

  return (bin_path,lib_path,inc_path)



CUDASuffixes = ['.cu']

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
CUDAScanner = SCons.Scanner.C.CScanner()

def add_common_nvcc_variables(env):
  """
  Add underlying common "NVIDIA CUDA compiler" variables that
  are used by multiple builders.
  """

  # "NVCC common command line"
  if not env.has_key('_NVCCCOMCOM'):
    # nvcc needs '-I' prepended before each include path, regardless of platform
    env['_NVCCWRAPCPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__)}'
    # prepend -Xcompiler before each flag
    env['_NVCCWRAPCFLAGS'] =     '${_concat("-Xcompiler ", CFLAGS,     "", __env__)}'
    env['_NVCCWRAPSHCFLAGS'] =   '${_concat("-Xcompiler ", SHCFLAGS,   "", __env__)}'
    env['_NVCCWRAPCCFLAGS'] =   '${_concat("-Xcompiler ", CCFLAGS,   "", __env__)}'
    env['_NVCCWRAPSHCCFLAGS'] = '${_concat("", SHCCFLAGS, "", __env__)}'
    # assemble the common command line
    env['_NVCCCOMCOM'] = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__)} $_CPPDEFFLAGS $_NVCCWRAPCPPPATH'

def generate(env):
  """
  Add Builders and construction variables for CUDA compilers to an Environment.
  """

  # create a builder that makes PTX files from .cu files
  ptx_builder = SCons.Builder.Builder(action = '$NVCC -ptx $NVCCFLAGS $_NVCCWRAPCFLAGS $NVCCWRAPCCFLAGS $_NVCCCOMCOM $SOURCES -o $TARGET',
                                      emitter = {},
                                      suffix = '.ptx',
                                      src_suffix = CUDASuffixes)
  env['BUILDERS']['PTXFile'] = ptx_builder

  # create builders that make static & shared objects from .cu files
  static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

  for suffix in CUDASuffixes:
    # Add this suffix to the list of things buildable by Object
    static_obj.add_action('$CUDAFILESUFFIX', '$NVCCCOM')
    shared_obj.add_action('$CUDAFILESUFFIX', '$SHNVCCCOM')
    static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
    shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)

    # Add this suffix to the list of things scannable
    SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

  add_common_nvcc_variables(env)

  # set the "CUDA Compiler Command" environment variable
  # windows is picky about getting the full filename of the executable
  if os.name == 'nt':
    env['NVCC'] = 'nvcc.exe'
    env['SHNVCC'] = 'nvcc.exe'
  else:
    env['NVCC'] = 'nvcc'
    env['SHNVCC'] = 'nvcc'
  
  # set the include path, and pass both c compiler flags and c++ compiler flags
  env['NVCCFLAGS'] = SCons.Util.CLVar('')
  env['SHNVCCFLAGS'] = SCons.Util.CLVar('') + ' -shared'
  
  # 'NVCC Command'
  env['NVCCCOM']   = '$NVCC -o $TARGET -c $NVCCFLAGS $_NVCCWRAPCFLAGS $NVCCWRAPCCFLAGS $_NVCCCOMCOM $SOURCES'
  env['SHNVCCCOM'] = '$SHNVCC -o $TARGET -c $SHNVCCFLAGS $_NVCCWRAPSHCFLAGS $_NVCCWRAPSHCCFLAGS $_NVCCCOMCOM $SOURCES'
  
  # the suffix of CUDA source files is '.cu'
  env['CUDAFILESUFFIX'] = '.cu'

  # XXX add code to generate builders for other miscellaneous
  # CUDA files here, such as .gpu, etc.

  # XXX intelligently detect location of nvcc and cuda libraries here
  (bin_path,lib_path,inc_path) = get_cuda_paths()
  env.AppendUnique(LIBPATH=[lib_path])
  env.AppendUnique(CPPPATH=[inc_path])
  env.PrependENVPath('PATH', bin_path)

def exists(env):
  return env.Detect('nvcc')

