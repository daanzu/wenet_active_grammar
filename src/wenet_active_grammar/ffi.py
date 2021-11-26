#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

"""
FFI classes for Wenet
"""

import os, re, sys

from cffi import FFI

if sys.platform.startswith('win'): _platform = 'windows'
elif sys.platform.startswith('linux'): _platform = 'linux'
elif sys.platform.startswith('darwin'): _platform = 'macos'
else: raise Exception("unknown sys.platform")

_ffi = FFI()
_library_directory_path = os.path.dirname(os.path.abspath(__file__))
_library_binary_path = os.path.join(_library_directory_path,
    dict(windows='wenet_stt_lib.dll', linux='libwenet_stt_lib.so', macos='libwenet_stt_lib.dylib')[_platform])
_c_source_ignore_regex = re.compile(r'(\b(extern|WENET_STT_API)\b)|("C")|(//.*$)', re.MULTILINE)  # Pattern for extraneous stuff to be removed

def encode(text):
    """ For C interop: encode unicode text -> binary utf-8. """
    return text.encode('utf-8')
def decode(binary):
    """ For C interop: decode binary utf-8 -> unicode text. """
    return binary.decode('utf-8')

class FFIObject(object):

    def __init__(self):
        self.init_ffi()

    @classmethod
    def init_ffi(cls):
        cls._lib = _ffi.init_once(cls._init_ffi, cls.__name__ + '._init_ffi')

    @classmethod
    def _init_ffi(cls):
        _ffi.cdef(_c_source_ignore_regex.sub(' ', cls._library_header_text))
        # On Windows, we need to temporarily prepend the PATH with the directory containing the DLLs to ensure that the DLLs are found.
        if _platform == 'windows':
            os.environ['PATH'] = os.pathsep.join([_library_directory_path, os.environ['PATH']])
        try:
            return _ffi.dlopen(_library_binary_path)
        finally:
            if _platform == 'windows':
                os.environ['PATH'] = os.pathsep.join(os.environ['PATH'].split(os.pathsep)[1:])
