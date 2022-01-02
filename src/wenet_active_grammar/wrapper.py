#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import json, os, time

import numpy as np

from . import _log, WenetError
from .ffi import FFIObject, _ffi, decode, encode, decode as de, encode as en
from .wfst import NativeWFST

_log = _log.getChild('wrapper')
_log_library = _log.getChild('library')


########################################################################################################################

class WenetSTTModel(FFIObject):

    _library_header_text = """
        WENET_STT_API void *wenet_stt__construct_model(const char *config_json_cstr);
        WENET_STT_API bool wenet_stt__destruct_model(void *model_vp);
        WENET_STT_API bool wenet_stt__decode_utterance(void *model_vp, float *wav_samples, int32_t wav_samples_len, char *text, int32_t text_max_len);
    """

    def __init__(self, config):
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        assert 'model_path' in config
        if not os.path.exists(config['model_path']):
            raise FileNotFoundError("model_path does not exist")
        assert 'dict_path' in config
        if not os.path.exists(config['dict_path']):
            raise FileNotFoundError("dict_path does not exist")

        super().__init__()
        result = self._lib.wenet_stt__construct_model(encode(json.dumps(config)))
        if result == _ffi.NULL:
            raise WenetError("wenet_stt__construct_model failed")
        self._model = result

    def __del__(self):
        if hasattr(self, '_model'):
            result = self._lib.wenet_stt__destruct_model(self._model)
            if not result:
                raise WenetError("wenet_stt__destruct_model failed")

    @classmethod
    def build_config(cls, model_dir=None, config=None):
        if config is None:
            config = dict()
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        config = config.copy()
        if model_dir is not None:
            config['model_path'] = os.path.join(model_dir, 'final.zip')
            config['dict_path'] = os.path.join(model_dir, 'words.txt')
        return config

    @classmethod
    def download_model_if_not_exists(cls, name, parent_dir='.', verbose=False):
        if os.path.exists(os.path.join(parent_dir, name)):
            return False
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        download_model(name, parent_dir=parent_dir, verbose=verbose)
        return True

    def decode(self, wav_samples, text_max_len=1024):
        if not isinstance(wav_samples, np.ndarray): wav_samples = np.frombuffer(wav_samples, np.int16)
        wav_samples = wav_samples.astype(np.float32)
        wav_samples_char = _ffi.from_buffer(wav_samples)
        wav_samples_float = _ffi.cast('float *', wav_samples_char)
        text_p = _ffi.new('char[]', text_max_len)

        result = self._lib.wenet_stt__decode_utterance(self._model, wav_samples_float, len(wav_samples), text_p, text_max_len)
        if not result:
            raise WenetError("wenet_stt__decode_utterance failed")

        text = decode(_ffi.string(text_p))
        if len(text) >= (text_max_len - 1):
            raise WenetError("text may be too long")
        return text.strip()

class WenetSTTDecoder(FFIObject):

    _library_header_text = """
        WENET_STT_API void *wenet_stt__construct_decoder(void *model_vp);
        WENET_STT_API bool wenet_stt__destruct_decoder(void *decoder_vp);
        WENET_STT_API bool wenet_stt__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize);
        WENET_STT_API bool wenet_stt__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p);
        WENET_STT_API bool wenet_stt__reset(void *decoder_vp);
    """

    def __init__(self, model):
        if not isinstance(model, WenetSTTModel):
            raise TypeError("model must be a WenetSTTModel")

        super().__init__()
        result = self._lib.wenet_stt__construct_decoder(model._model)
        if result == _ffi.NULL:
            raise WenetError("wenet_stt__construct_decoder failed")
        self._decoder = result

    def __del__(self):
        if hasattr(self, '_decoder'):
            result = self._lib.wenet_stt__destruct_decoder(self._decoder)
            if not result:
                raise WenetError("wenet_stt__destruct_decoder failed")

    def decode(self, wav_samples, finalize):
        if not isinstance(wav_samples, np.ndarray): wav_samples = np.frombuffer(wav_samples, np.int16)
        wav_samples = wav_samples.astype(np.float32)
        wav_samples_char = _ffi.from_buffer(wav_samples)
        wav_samples_float = _ffi.cast('float *', wav_samples_char)
        finalize = bool(finalize)

        result = self._lib.wenet_stt__decode(self._decoder, wav_samples_float, len(wav_samples), finalize)
        if not result:
            raise WenetError("wenet_stt__decode failed")

    def get_result(self, final=None, text_max_len=1024):
        text_p = _ffi.new('char[]', text_max_len)
        result_final_p = _ffi.new('bool *')

        while True:
            result = self._lib.wenet_stt__get_result(self._decoder, text_p, text_max_len, result_final_p)
            if not result:
                raise WenetError("wenet_stt__get_result failed")
            result_final = bool(result_final_p[0])
            if not final or result_final:
                break
            time.sleep(0.01)

        text = decode(_ffi.string(text_p))
        if len(text) >= (text_max_len - 1):
            raise WenetError("text may be too long")
        return text.strip(), result_final

    def reset(self):
        result = self._lib.wenet_stt__reset(self._decoder)
        if not result:
            raise WenetError("wenet_stt__reset failed")

class WenetAGDecoder(FFIObject):

    _library_header_text = """
        WENET_STT_API void *wenet_ag__construct_decoder(void *model_vp);
        WENET_STT_API bool wenet_ag__destruct_decoder(void *decoder_vp);
        WENET_STT_API bool wenet_ag__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize);
        WENET_STT_API bool wenet_ag__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p, int32_t *rule_number_p);
        WENET_STT_API bool wenet_ag__reset(void *decoder_vp);
        WENET_STT_API bool wenet_ag__set_grammars_activity(void *decoder_vp, bool *grammars_activity_cp, int32_t grammars_activity_cp_size);
        WENET_STT_API int32_t wenet_ag__add_grammar_fst(void *decoder_vp, void *grammar_fst_vp);
        WENET_STT_API bool wenet_ag__reload_grammar_fst(void *decoder_vp, int32_t grammar_fst_index, void *grammar_fst_vp);
        WENET_STT_API bool wenet_ag__remove_grammar_fst(void *decoder_vp, int32_t grammar_fst_index);
    """

    def __init__(self, model):
        if not isinstance(model, WenetSTTModel):
            raise TypeError("model must be a WenetSTTModel")
        # _log.debug("WenetAGDecoder initialized in process %s", os.getpid())

        super().__init__()
        result = self._lib.wenet_ag__construct_decoder(model._model)
        if result == _ffi.NULL:
            raise WenetError("wenet_ag__construct_decoder failed")
        self._decoder = result
        self.num_grammars = 0

    def __del__(self):
        if hasattr(self, '_decoder'):
            result = self._lib.wenet_ag__destruct_decoder(self._decoder)
            if not result:
                raise WenetError("wenet_ag__destruct_decoder failed")

    def decode(self, wav_samples, finalize):
        if not isinstance(wav_samples, np.ndarray): wav_samples = np.frombuffer(wav_samples, np.int16)
        wav_samples = wav_samples.astype(np.float32)
        wav_samples_char = _ffi.from_buffer(wav_samples)
        wav_samples_float = _ffi.cast('float *', wav_samples_char)
        finalize = bool(finalize)

        result = self._lib.wenet_ag__decode(self._decoder, wav_samples_float, len(wav_samples), finalize)
        if not result:
            raise WenetError("wenet_ag__decode failed")

    def get_result(self, final=None, text_max_len=1024):
        text_p = _ffi.new('char[]', text_max_len)
        result_final_p = _ffi.new('bool *')
        rule_number_p = _ffi.new('int32_t *')

        while True:
            result = self._lib.wenet_ag__get_result(self._decoder, text_p, text_max_len, result_final_p, rule_number_p)
            if not result:
                raise WenetError("wenet_ag__get_result failed")
            result_final = bool(result_final_p[0])
            rule_number = int(rule_number_p[0])
            if not final or result_final:
                break
            time.sleep(0.01)

        text = decode(_ffi.string(text_p))
        if len(text) >= (text_max_len - 1):
            raise WenetError("text may be too long")
        return text.strip(), result_final, rule_number

    def reset(self):
        result = self._lib.wenet_ag__reset(self._decoder)
        if not result:
            raise WenetError("wenet_ag__reset failed")

    def set_grammars_activity(self, grammars_activity):
        # _log.log(5, "set_grammars_activity %s", ''.join('1' if a else '0' for a in grammars_activity))
        if len(grammars_activity) != self.num_grammars:
            _log.error("wrong len(grammars_activity) = %d != %d = num_grammars", len(grammars_activity), self.num_grammars)
        result = self._lib.wenet_ag__set_grammars_activity(self._decoder, grammars_activity, len(grammars_activity))
        if not result:
            raise WenetError("wenet_ag__set_grammars_activity failed")

    def add_grammar_fst(self, grammar_fst):
        assert isinstance(grammar_fst, NativeWFST)
        _log.log(8, "%s: adding grammar_fst: %r", self, grammar_fst)
        grammar_fst_index = self._lib.wenet_ag__add_grammar_fst(self._decoder, grammar_fst.native_obj)
        if grammar_fst_index < 0:
            raise WenetError("wenet_ag__add_grammar_fst failed: %r" % grammar_fst)
        assert grammar_fst_index == self.num_grammars, "add_grammar_fst allocated invalid grammar_fst_index"
        self.num_grammars += 1
        return grammar_fst_index

    def reload_grammar_fst(self, grammar_fst_index, grammar_fst):
        assert isinstance(grammar_fst, NativeWFST)
        _log.debug("%s: reloading grammar_fst_index: #%s %r", self, grammar_fst_index, grammar_fst)
        result = self._lib.wenet_ag__reload_grammar_fst(self._decoder, grammar_fst_index, grammar_fst.native_obj)
        if not result:
            raise WenetError("wenet_ag__reload_grammar_fst failed: #%s %r" % (grammar_fst_index, grammar_fst))

    def remove_grammar_fst(self, grammar_fst_index):
        _log.debug("%s: removing grammar_fst_index: %s", self, grammar_fst_index)
        result = self._lib.wenet_ag__remove_grammar_fst(self._decoder, grammar_fst_index)
        if not result:
            raise WenetError("wenet_ag__remove_grammar_fst failed: #%s" % grammar_fst_index)
        self.num_grammars -= 1
