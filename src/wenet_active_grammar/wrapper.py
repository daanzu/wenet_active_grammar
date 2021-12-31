#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import json, os, re, sys, time

import numpy as np

from . import _log, WenetError
from .ffi import FFIObject, _ffi, decode, encode, decode as de, encode as en
from .utils import clock, find_file, show_donation_message, symbol_table_lookup
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
            raise Exception("wenet_stt__construct_model failed")
        self._model = result

    def __del__(self):
        if hasattr(self, '_model'):
            result = self._lib.wenet_stt__destruct_model(self._model)
            if not result:
                raise Exception("wenet_stt__destruct_model failed")

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
            raise Exception("wenet_stt__decode_utterance failed")

        text = decode(_ffi.string(text_p))
        if len(text) >= (text_max_len - 1):
            raise Exception("text may be too long")
        return text.strip()

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
            raise Exception("wenet_ag__construct_decoder failed")
        self._decoder = result
        self.num_grammars = 0

    def __del__(self):
        if hasattr(self, '_decoder'):
            result = self._lib.wenet_ag__destruct_decoder(self._decoder)
            if not result:
                raise Exception("wenet_ag__destruct_decoder failed")

    def decode(self, wav_samples, finalize):
        if not isinstance(wav_samples, np.ndarray): wav_samples = np.frombuffer(wav_samples, np.int16)
        wav_samples = wav_samples.astype(np.float32)
        wav_samples_char = _ffi.from_buffer(wav_samples)
        wav_samples_float = _ffi.cast('float *', wav_samples_char)
        finalize = bool(finalize)

        result = self._lib.wenet_ag__decode(self._decoder, wav_samples_float, len(wav_samples), finalize)
        if not result:
            raise Exception("wenet_ag__decode failed")

    def get_result(self, final=None, text_max_len=1024):
        text_p = _ffi.new('char[]', text_max_len)
        result_final_p = _ffi.new('bool *')
        rule_number_p = _ffi.new('int32_t *')

        while True:
            result = self._lib.wenet_ag__get_result(self._decoder, text_p, text_max_len, result_final_p, rule_number_p)
            if not result:
                raise Exception("wenet_ag__get_result failed")
            result_final = bool(result_final_p[0])
            rule_number = int(rule_number_p[0])
            if not final or result_final:
                break
            time.sleep(0.01)

        text = decode(_ffi.string(text_p))
        if len(text) >= (text_max_len - 1):
            raise Exception("text may be too long")
        return text.strip(), result_final, rule_number

    def reset(self):
        result = self._lib.wenet_ag__reset(self._decoder)
        if not result:
            raise Exception("wenet_ag__reset failed")

    def set_grammars_activity(self, grammars_activity):
        # _log.log(5, "set_grammars_activity %s", ''.join('1' if a else '0' for a in grammars_activity))
        if len(grammars_activity) != self.num_grammars:
            _log.error("wrong len(grammars_activity) = %d != %d = num_grammars", len(grammars_activity), self.num_grammars)
        result = self._lib.wenet_ag__set_grammars_activity(self._decoder, grammars_activity, len(grammars_activity))
        if not result:
            raise WenetError("wenet_ag__set_grammars_activity failed")

    def add_grammar_fst(self, grammar_fst):
        _log.log(8, "%s: adding grammar_fst: %r", self, grammar_fst)
        grammar_fst_index = self._lib.wenet_ag__add_grammar_fst(self._decoder, grammar_fst.native_obj)
        if grammar_fst_index < 0:
            raise WenetError("wenet_ag__add_grammar_fst failed: %r" % grammar_fst)
        assert grammar_fst_index == self.num_grammars, "add_grammar_fst allocated invalid grammar_fst_index"
        self.num_grammars += 1
        return grammar_fst_index

    def reload_grammar_fst(self, grammar_fst_index, grammar_fst):
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









########################################################################################################################

class WenetDecoderBase(FFIObject):

    def __init__(self):
        super(WenetDecoderBase, self).__init__()

        show_donation_message()

        self.sample_rate = 16000
        self.num_channels = 1
        self.bytes_per_wenet_frame = self.wenet_frame_num_to_audio_bytes(1)

        self._reset_decode_time()

    def _reset_decode_time(self):
        self._decode_time = 0
        self._decode_real_time = 0
        self._decode_times = []

    def _start_decode_time(self, num_frames):
        self.decode_start_time = clock()
        self._decode_real_time += 1000.0 * num_frames / self.sample_rate

    def _stop_decode_time(self, finalize=False):
        this = (clock() - self.decode_start_time) * 1000.0
        self._decode_time += this
        self._decode_times.append(this)
        if finalize:
            rtf = 1.0 * self._decode_time / self._decode_real_time if self._decode_real_time != 0 else float('nan')
            pct = 100.0 * this / self._decode_time if self._decode_time != 0 else 100
            _log.log(15, "decoded at %.2f RTF, for %d ms audio, spending %d ms, of which %d ms (%.0f%%) in finalization",
                rtf, self._decode_real_time, self._decode_time, this, pct)
            _log.log(13, "    decode times: %s", ' '.join("%d" % t for t in self._decode_times))
            self._reset_decode_time()

    def wenet_frame_num_to_audio_bytes(self, wenet_frame_num):
        wenet_frame_length_ms = 30
        sample_size_bytes = 2 * self.num_channels
        return int(wenet_frame_num * wenet_frame_length_ms * self.sample_rate / 1000 * sample_size_bytes)

    def audio_bytes_to_s(self, audio_bytes):
        sample_size_bytes = 2 * self.num_channels
        return 1.0 * audio_bytes // sample_size_bytes / self.sample_rate


########################################################################################################################

class WenetNNet3Decoder(WenetDecoderBase):
    """ Abstract base class for nnet3 decoders. """

    _library_header_text = """
        WENET_STT_API bool nnet3_base__load_lexicon(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp);
        WENET_STT_API bool nnet3_base__save_adaptation_state(void* model_vp);
        WENET_STT_API bool nnet3_base__reset_adaptation_state(void* model_vp);
        WENET_STT_API bool nnet3_base__get_word_align(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
        WENET_STT_API bool nnet3_base__decode(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);
        WENET_STT_API bool nnet3_base__get_output(void* model_vp, char* output, int32_t output_max_length,
                float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p);
        WENET_STT_API bool nnet3_base__set_lm_prime_text(void* model_vp, char* prime_cp);
    """

    def __init__(self, model_dir, tmp_dir, words_file=None, word_align_lexicon_file=None, max_num_rules=None, save_adaptation_state=False):
        super(WenetNNet3Decoder, self).__init__()

        model_dir = os.path.normpath(model_dir)
        if words_file is None: words_file = find_file(model_dir, 'words.txt')
        if word_align_lexicon_file is None: word_align_lexicon_file = find_file(model_dir, 'align_lexicon.int', required=False)
        mfcc_conf_file = find_file(model_dir, 'mfcc_hires.conf')
        if mfcc_conf_file is None: mfcc_conf_file = find_file(model_dir, 'mfcc.conf')  # FIXME: warning?
        model_file = find_file(model_dir, 'final.mdl')

        self.model_dir = model_dir
        self.words_file = os.path.normpath(words_file)
        self.word_align_lexicon_file = os.path.normpath(word_align_lexicon_file) if word_align_lexicon_file is not None else None
        self.mfcc_conf_file = os.path.normpath(mfcc_conf_file)
        self.model_file = os.path.normpath(model_file)
        self.ie_config = self._read_ie_conf_file(model_dir, find_file(model_dir, 'ivector_extractor.conf'))
        self.verbosity = (10 - _log_library.getEffectiveLevel()) if _log_library.isEnabledFor(10) else -1
        self.max_num_rules = int(max_num_rules) if max_num_rules is not None else None
        self._saving_adaptation_state = save_adaptation_state

        self.config_dict = {
            'model_dir': self.model_dir,
            'mfcc_config_filename': self.mfcc_conf_file,
            'ivector_extraction_config_json': self.ie_config,
            'model_filename': self.model_file,
            'word_syms_filename': self.words_file,
            'word_align_lexicon_filename': self.word_align_lexicon_file or '',
            }
        if self.max_num_rules is not None: self.config_dict.update(max_num_rules=self.max_num_rules)

    def _read_ie_conf_file(self, model_dir, old_filename, search=True):
        """ Read ivector_extractor.conf file, converting relative paths to absolute paths for current configuration, returning dict of config. """
        options_with_path = {
            '--splice-config':      'conf/splice.conf',
            '--cmvn-config':        'conf/online_cmvn.conf',
            '--lda-matrix':         'ivector_extractor/final.mat',
            '--global-cmvn-stats':  'ivector_extractor/global_cmvn.stats',
            '--diag-ubm':           'ivector_extractor/final.dubm',
            '--ivector-extractor':  'ivector_extractor/final.ie',
        }
        def convert_path(key, value):
            if not search:
                return os.path.join(model_dir, options_with_path[key])
            else:
                return find_file(model_dir, os.path.basename(options_with_path[key]), required=True)
        options_converters = {
            '--splice-config':          convert_path,
            '--cmvn-config':            convert_path,
            '--lda-matrix':             convert_path,
            '--global-cmvn-stats':      convert_path,
            '--diag-ubm':               convert_path,
            '--ivector-extractor':      convert_path,
            '--ivector-period':         lambda key, value: (float(value) if '.' in value else int(value)),
            '--max-count':              lambda key, value: (float(value) if '.' in value else int(value)),
            '--max-remembered-frames':  lambda key, value: (float(value) if '.' in value else int(value)),
            '--min-post':               lambda key, value: (float(value) if '.' in value else int(value)),
            '--num-gselect':            lambda key, value: (float(value) if '.' in value else int(value)),
            '--posterior-scale':        lambda key, value: (float(value) if '.' in value else int(value)),
            '--online-cmvn-iextractor': lambda key, value: (True if value in ['true'] else False),
        }
        config = dict()
        with open(old_filename, 'r', encoding='utf-8') as old_file:
            for line in old_file:
                key, value = line.strip().split('=', 1)
                value = options_converters[key](key, value)
                assert key.startswith('--')
                key = key[2:]
                config[key] = value
        return config

    saving_adaptation_state = property(lambda self: self._saving_adaptation_state, doc="Whether currently to save updated adaptation state at end of utterance")
    @saving_adaptation_state.setter
    def saving_adaptation_state(self, value): self._saving_adaptation_state = value

    def load_lexicon(self, words_file=None, word_align_lexicon_file=None):
        """ Only necessary when you update the lexicon after initialization. """
        if words_file is None: words_file = self.words_file
        if word_align_lexicon_file is None: word_align_lexicon_file = self.word_align_lexicon_file
        result = self._lib.nnet3_base__load_lexicon(self._model, en(words_file), en(word_align_lexicon_file))
        if not result:
            raise WenetError("error loading lexicon (%r, %r)" % (words_file, word_align_lexicon_file))

    def save_adaptation_state(self):
        result = self._lib.nnet3_base__save_adaptation_state(self._model)
        if not result:
            raise WenetError("save_adaptation_state error")

    def reset_adaptation_state(self):
        result = self._lib.nnet3_base__reset_adaptation_state(self._model)
        if not result:
            raise WenetError("reset_adaptation_state error")

    def get_output(self, output_max_length=4*1024):
        output_p = _ffi.new('char[]', output_max_length)
        likelihood_p = _ffi.new('float *')
        am_score_p = _ffi.new('float *')
        lm_score_p = _ffi.new('float *')
        confidence_p = _ffi.new('float *')
        expected_error_rate_p = _ffi.new('float *')
        result = self._lib.nnet3_base__get_output(self._model, output_p, output_max_length, likelihood_p, am_score_p, lm_score_p, confidence_p, expected_error_rate_p)
        if not result:
            raise WenetError("get_output error")
        output_str = de(_ffi.string(output_p))
        info = {
            'likelihood': likelihood_p[0],
            'am_score': am_score_p[0],
            'lm_score': lm_score_p[0],
            'confidence': confidence_p[0],
            'expected_error_rate': expected_error_rate_p[0],
        }
        _log.log(7, "get_output: %r %s", output_str, info)
        return output_str, info

    def get_word_align(self, output):
        """Returns tuple of tuples: words (including nonterminals but not eps), each's time (in bytes), and each's length (in bytes)."""
        words = output.split()
        num_words = len(words)
        wenet_frame_times_p = _ffi.new('int32_t[]', num_words)
        wenet_frame_lengths_p = _ffi.new('int32_t[]', num_words)
        result = self._lib.nnet3_base__get_word_align(self._model, wenet_frame_times_p, wenet_frame_lengths_p, num_words)
        if not result:
            raise WenetError("get_word_align error")
        times = [wenet_frame_num * self.bytes_per_wenet_frame for wenet_frame_num in wenet_frame_times_p]
        lengths = [wenet_frame_num * self.bytes_per_wenet_frame for wenet_frame_num in wenet_frame_lengths_p]
        return tuple(zip(words, times, lengths))

    def set_lm_prime_text(self, prime_text):
        prime_text = prime_text.strip()
        result = self._lib.nnet3_base__set_lm_prime_text(self._model, en(prime_text))
        if not result:
            raise WenetError("error setting prime text %r" % prime_text)


########################################################################################################################

class WenetPlainNNet3Decoder(WenetNNet3Decoder):

    _library_header_text = WenetNNet3Decoder._library_header_text + """
        WENET_STT_API void* nnet3_plain__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
        WENET_STT_API bool nnet3_plain__destruct(void* model_vp);
        WENET_STT_API bool nnet3_plain__decode(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);
    """

    def __init__(self, fst_file=None, config=None, **kwargs):
        super(WenetPlainNNet3Decoder, self).__init__(**kwargs)

        if fst_file is None: fst_file = find_file(self.model_dir, defaults.DEFAULT_PLAIN_DICTATION_HCLG_FST_FILENAME, required=True)
        fst_file = os.path.normpath(fst_file)

        self.config_dict.update({
            'decode_fst_filename': fst_file,
            })
        if config: self.config_dict.update(config)

        _log.debug("config_dict: %s", self.config_dict)
        self._model = self._lib.nnet3_plain__construct(en(self.model_dir), en(json.dumps(self.config_dict)), self.verbosity)
        if not self._model: raise WenetError("failed nnet3_plain__construct")

    def destroy(self):
        if self._model:
            result = self._lib.nnet3_plain__destruct(self._model)
            if not result:
                raise WenetError("failed nnet3_plain__destruct")
            self._model = None

    def decode(self, frames, finalize):
        """Continue decoding with given new audio data."""
        if not isinstance(frames, np.ndarray): frames = np.frombuffer(frames, np.int16)
        frames = frames.astype(np.float32)
        frames_char = _ffi.from_buffer(frames)
        frames_float = _ffi.cast('float *', frames_char)

        self._start_decode_time(len(frames))
        result = self._lib.nnet3_plain__decode(self._model, self.sample_rate, len(frames), frames_float, finalize, self._saving_adaptation_state)
        self._stop_decode_time(finalize)

        if not result:
            raise WenetError("decoding error")
        return finalize


########################################################################################################################

class WenetAgfNNet3Decoder(WenetNNet3Decoder):
    """docstring for WenetAgfNNet3Decoder"""

    _library_header_text = WenetNNet3Decoder._library_header_text + """
        WENET_STT_API void* nnet3_agf__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
        WENET_STT_API bool nnet3_agf__destruct(void* model_vp);
        WENET_STT_API int32_t nnet3_agf__add_grammar_fst(void* model_vp, void* grammar_fst_cp);
        WENET_STT_API int32_t nnet3_agf__add_grammar_fst_file(void* model_vp, char* grammar_fst_filename_cp);
        WENET_STT_API bool nnet3_agf__reload_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp);
        WENET_STT_API bool nnet3_agf__reload_grammar_fst_file(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp);
        WENET_STT_API bool nnet3_agf__remove_grammar_fst(void* model_vp, int32_t grammar_fst_index);
        WENET_STT_API bool nnet3_agf__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
            bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
    """

    def __init__(self, *, top_fst=None, dictation_fst_file=None, config=None, **kwargs):
        super(WenetAgfNNet3Decoder, self).__init__(**kwargs)

        phones_file = find_file(self.model_dir, 'phones.txt')
        nonterm_phones_offset = symbol_table_lookup(phones_file, '#nonterm_bos')
        if nonterm_phones_offset is None:
            raise WenetError("cannot find #nonterm_bos symbol in phones.txt")
        rules_phones_offset = symbol_table_lookup(phones_file, '#nonterm:rule0')
        if rules_phones_offset is None:
            raise WenetError("cannot find #nonterm:rule0 symbol in phones.txt")
        dictation_phones_offset = symbol_table_lookup(phones_file, '#nonterm:dictation')
        if dictation_phones_offset is None:
            raise WenetError("cannot find #nonterm:dictation symbol in phones.txt")

        self.config_dict.update({
            'nonterm_phones_offset': nonterm_phones_offset,
            'rules_phones_offset': rules_phones_offset,
            'dictation_phones_offset': dictation_phones_offset,
            'dictation_fst_filename': os.path.normpath(dictation_fst_file) if dictation_fst_file is not None else '',
            })
        if isinstance(top_fst, NativeWFST): self.config_dict.update({'top_fst': int(_ffi.cast("uint64_t", top_fst.compiled_native_obj))})
        elif isinstance(top_fst, str): self.config_dict.update({'top_fst_filename': os.path.normpath(top_fst)})
        else: raise WenetError("unrecognized top_fst type")
        if config: self.config_dict.update(config)

        _log.debug("config_dict: %s", self.config_dict)
        self._model = self._lib.nnet3_agf__construct(en(self.model_dir), en(json.dumps(self.config_dict)), self.verbosity)
        if not self._model: raise WenetError("failed nnet3_agf__construct")
        self.num_grammars = 0

    def destroy(self):
        if self._model:
            result = self._lib.nnet3_agf__destruct(self._model)
            if not result:
                raise WenetError("failed nnet3_agf__destruct")
            self._model = None

    def add_grammar_fst(self, grammar_fst):
        _log.log(8, "%s: adding grammar_fst: %r", self, grammar_fst)
        if isinstance(grammar_fst, NativeWFST):
            grammar_fst_index = self._lib.nnet3_agf__add_grammar_fst(self._model, grammar_fst.compiled_native_obj)
        elif isinstance(grammar_fst, str):
            grammar_fst_index = self._lib.nnet3_agf__add_grammar_fst_file(self._model, en(os.path.normpath(grammar_fst)))
        else: raise WenetError("unrecognized grammar_fst type")
        if grammar_fst_index < 0:
            raise WenetError("error adding grammar %r" % grammar_fst)
        assert grammar_fst_index == self.num_grammars, "add_grammar_fst allocated invalid grammar_fst_index"
        self.num_grammars += 1
        return grammar_fst_index

    def reload_grammar_fst(self, grammar_fst_index, grammar_fst):
        _log.debug("%s: reloading grammar_fst_index: #%s %r", self, grammar_fst_index, grammar_fst)
        if isinstance(grammar_fst, NativeWFST):
            result = self._lib.nnet3_agf__reload_grammar_fst(self._model, grammar_fst_index, grammar_fst.compiled_native_obj)
        elif isinstance(grammar_fst, str):
            result = self._lib.nnet3_agf__reload_grammar_fst_file(self._model, grammar_fst_index, en(os.path.normpath(grammar_fst)))
        else: raise WenetError("unrecognized grammar_fst type")
        if not result:
            raise WenetError("error reloading grammar #%s %r" % (grammar_fst_index, grammar_fst))

    def remove_grammar_fst(self, grammar_fst_index):
        _log.debug("%s: removing grammar_fst_index: %s", self, grammar_fst_index)
        result = self._lib.nnet3_agf__remove_grammar_fst(self._model, grammar_fst_index)
        if not result:
            raise WenetError("error removing grammar #%s" % grammar_fst_index)
        self.num_grammars -= 1

    def decode(self, frames, finalize, grammars_activity=None):
        """Continue decoding with given new audio data."""
        # grammars_activity = [True] * self.num_grammars
        # grammars_activity = np.random.choice([True, False], len(grammars_activity)).tolist(); print grammars_activity; time.sleep(5)
        if grammars_activity is None:
            grammars_activity = []
        else:
            # Start of utterance
            _log.log(5, "decode: grammars_activity = %s", ''.join('1' if a else '0' for a in grammars_activity))
            if len(grammars_activity) != self.num_grammars:
                _log.error("wrong len(grammars_activity) = %d != %d = num_grammars" % (len(grammars_activity), self.num_grammars))

        if not isinstance(frames, np.ndarray): frames = np.frombuffer(frames, np.int16)
        frames = frames.astype(np.float32)
        frames_char = _ffi.from_buffer(frames)
        frames_float = _ffi.cast('float *', frames_char)

        self._start_decode_time(len(frames))
        result = self._lib.nnet3_agf__decode(self._model, self.sample_rate, len(frames), frames_float, finalize,
            grammars_activity, len(grammars_activity), self._saving_adaptation_state)
        self._stop_decode_time(finalize)

        if not result:
            raise WenetError("decoding error")
        return finalize


########################################################################################################################

class WenetAgfCompiler(FFIObject):

    _library_header_text = """
        WENET_STT_API void* nnet3_agf__construct_compiler(char* config_str_cp);
        WENET_STT_API bool nnet3_agf__destruct_compiler(void* compiler_vp);
        WENET_STT_API void* nnet3_agf__compile_graph(void* compiler_vp, char* config_str_cp, void* grammar_fst_cp, bool return_graph);
        WENET_STT_API void* nnet3_agf__compile_graph_text(void* compiler_vp, char* config_str_cp, char* grammar_fst_text_cp, bool return_graph);
        WENET_STT_API void* nnet3_agf__compile_graph_file(void* compiler_vp, char* config_str_cp, char* grammar_fst_filename_cp, bool return_graph);
    """

    def __init__(self, config):
        super(WenetAgfCompiler, self).__init__()
        self._compiler = self._lib.nnet3_agf__construct_compiler(en(json.dumps(config)))
        if not self._compiler: raise WenetError("failed nnet3_agf__construct_compiler")

    def destroy(self):
        if self._compiler:
            result = self._lib.nnet3_agf__destruct_compiler(self._compiler)
            if not result:
                raise WenetError("failed nnet3_agf__destruct_compiler")
            self._compiler = None

    def compile_graph(self, config, grammar_fst=None, grammar_fst_text=None, grammar_fst_file=None, return_graph=False):
        if 1 != sum(int(g is not None) for g in [grammar_fst, grammar_fst_text, grammar_fst_file]):
            raise ValueError("must pass exactly one grammar")
        if grammar_fst is not None:
            _log.log(5, "compile_graph:\n    config=%r\n    grammar_fst=%r", config, grammar_fst)
            result = self._lib.nnet3_agf__compile_graph(self._compiler, en(json.dumps(config)), grammar_fst.native_obj, return_graph)
            return result
        if grammar_fst_text is not None:
            _log.log(5, "compile_graph:\n    config=%r\n    grammar_fst_text:\n%s", config, grammar_fst_text)
            result = self._lib.nnet3_agf__compile_graph_text(self._compiler, en(json.dumps(config)), en(grammar_fst_text), return_graph)
            return result
        if grammar_fst_file is not None:
            _log.log(5, "compile_graph:\n    config=%r\n    grammar_fst_file=%r", config, grammar_fst_file)
            result = self._lib.nnet3_agf__compile_graph_file(self._compiler, en(json.dumps(config)), en(grammar_fst_file), return_graph)
            return result


########################################################################################################################

class WenetLafNNet3Decoder(WenetNNet3Decoder):
    """docstring for WenetLafNNet3Decoder"""

    _library_header_text = WenetNNet3Decoder._library_header_text + """
        WENET_STT_API void* nnet3_laf__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
        WENET_STT_API bool nnet3_laf__destruct(void* model_vp);
        WENET_STT_API int32_t nnet3_laf__add_grammar_fst(void* model_vp, void* grammar_fst_cp);
        WENET_STT_API int32_t nnet3_laf__add_grammar_fst_text(void* model_vp, char* grammar_fst_cp);
        WENET_STT_API bool nnet3_laf__reload_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp);
        WENET_STT_API bool nnet3_laf__remove_grammar_fst(void* model_vp, int32_t grammar_fst_index);
        WENET_STT_API bool nnet3_laf__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
            bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
    """

    def __init__(self, dictation_fst_file=None, config=None, **kwargs):
        super(WenetLafNNet3Decoder, self).__init__(**kwargs)

        self.config_dict.update({
            'hcl_fst_filename': find_file(self.model_dir, 'HCLr.fst'),
            'disambig_tids_filename': find_file(self.model_dir, 'disambig_tid.int'),
            'relabel_ilabels_filename': find_file(self.model_dir, 'relabel_ilabels.int'),
            'word_syms_relabeled_filename': find_file(self.model_dir, 'words.relabeled.txt', required=True),
            'dictation_fst_filename': dictation_fst_file or '',
            })
        if config: self.config_dict.update(config)

        _log.debug("config_dict: %s", self.config_dict)
        self._model = self._lib.nnet3_laf__construct(en(self.model_dir), en(json.dumps(self.config_dict)), self.verbosity)
        if not self._model: raise WenetError("failed nnet3_laf__construct")
        self.num_grammars = 0

    def destroy(self):
        if self._model:
            result = self._lib.nnet3_laf__destruct(self._model)
            if not result:
                raise WenetError("failed nnet3_laf__destruct")
            self._model = None

    def add_grammar_fst(self, grammar_fst):
        _log.log(8, "%s: adding grammar_fst: %r", self, grammar_fst)
        grammar_fst_index = self._lib.nnet3_laf__add_grammar_fst(self._model, grammar_fst.native_obj)
        if grammar_fst_index < 0:
            raise WenetError("error adding grammar %r" % grammar_fst)
        assert grammar_fst_index == self.num_grammars, "add_grammar_fst allocated invalid grammar_fst_index"
        self.num_grammars += 1
        return grammar_fst_index

    def add_grammar_fst_text(self, grammar_fst_text):
        assert grammar_fst_text
        _log.log(8, "%s: adding grammar_fst_text: %r", self, grammar_fst_text[:512])
        grammar_fst_index = self._lib.nnet3_laf__add_grammar_fst_text(self._model, en(grammar_fst_text))
        if grammar_fst_index < 0:
            raise WenetError("error adding grammar %r" % grammar_fst_text[:512])
        assert grammar_fst_index == self.num_grammars, "add_grammar_fst allocated invalid grammar_fst_index"
        self.num_grammars += 1
        return grammar_fst_index

    def reload_grammar_fst(self, grammar_fst_index, grammar_fst):
        _log.debug("%s: reloading grammar_fst_index: #%s %r", self, grammar_fst_index, grammar_fst)
        result = self._lib.nnet3_laf__reload_grammar_fst(self._model, grammar_fst_index, grammar_fst.native_obj)
        if not result:
            raise WenetError("error reloading grammar #%s %r" % (grammar_fst_index, grammar_fst))

    def remove_grammar_fst(self, grammar_fst_index):
        _log.debug("%s: removing grammar_fst_index: %s", self, grammar_fst_index)
        result = self._lib.nnet3_laf__remove_grammar_fst(self._model, grammar_fst_index)
        if not result:
            raise WenetError("error removing grammar #%s" % grammar_fst_index)
        self.num_grammars -= 1

    def decode(self, frames, finalize, grammars_activity=None):
        """Continue decoding with given new audio data."""
        # grammars_activity = [True] * self.num_grammars
        # grammars_activity = np.random.choice([True, False], len(grammars_activity)).tolist(); print grammars_activity; time.sleep(5)
        if grammars_activity is None:
            grammars_activity = []
        else:
            # Start of utterance
            _log.log(5, "decode: grammars_activity = %s", ''.join('1' if a else '0' for a in grammars_activity))
            if len(grammars_activity) != self.num_grammars:
                _log.error("wrong len(grammars_activity) = %d != %d = num_grammars" % (len(grammars_activity), self.num_grammars))

        if not isinstance(frames, np.ndarray): frames = np.frombuffer(frames, np.int16)
        frames = frames.astype(np.float32)
        frames_char = _ffi.from_buffer(frames)
        frames_float = _ffi.cast('float *', frames_char)

        self._start_decode_time(len(frames))
        result = self._lib.nnet3_laf__decode(self._model, self.sample_rate, len(frames), frames_float, finalize,
            grammars_activity, len(grammars_activity), self._saving_adaptation_state)
        self._stop_decode_time(finalize)

        if not result:
            raise WenetError("decoding error")
        return finalize
