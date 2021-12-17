#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import collections, itertools, math

from six import iteritems, itervalues, text_type

from . import WenetError
from .utils import FSTFileCache


########################################################################################################################

from .ffi import FFIObject, _ffi, decode, encode

class NativeWFST(FFIObject):
    """
    WFST class, implemented in native code.
    Notes:
        * Weight (arc & state) is stored as raw probability, then normalized and converted to negative log likelihood/probability before export.
    """

    _library_header_text = """
        WENET_STT_API bool fst__init(int32_t eps_like_ilabels_len, int32_t eps_like_ilabels_cp[], int32_t silent_olabels_len, int32_t silent_olabels_cp[], int32_t wildcard_olabels_len, int32_t wildcard_olabels_cp[]);
        WENET_STT_API void* fst__construct();
        WENET_STT_API bool fst__destruct(void* fst_vp);
        WENET_STT_API int32_t fst__add_state(void* fst_vp, float weight, bool initial);
        WENET_STT_API bool fst__add_arc(void* fst_vp, int32_t src_state_id, int32_t dst_state_id, int32_t ilabel, int32_t olabel, float weight);
        WENET_STT_API bool fst__compute_md5(void* fst_vp, char* md5_cp, char* dependencies_seed_md5_cp);
        WENET_STT_API bool fst__has_path(void* fst_vp);
        WENET_STT_API bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state);
        WENET_STT_API bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len);
        WENET_STT_API void* fst__load_file(char* filename_cp);
        WENET_STT_API bool fst__write_file(void* fst_vp, char* filename_cp);
        WENET_STT_API bool fst__write_file_const(void* fst_vp, char* filename_cp);
        WENET_STT_API bool fst__print(void* fst_vp, char* filename_cp);
        WENET_STT_API void* fst__compile_text(char* fst_text_cp, char* isymbols_file_cp, char* osymbols_file_cp);
    """

    zero = float('inf')  # Weight of non-final states; a state is final if and only if its weight is not equal to self.zero
    one = 0.0
    eps = u'<eps>'
    eps_disambig = u'#0'
    eps_like_words = frozenset((eps, eps_disambig))
    silent_words = frozenset((eps, eps_disambig))  # u'!SIL'
    native = property(lambda self: True)

    @classmethod
    def init_class(cls, isymbol_table, wildcard_nonterms, osymbol_table=None):
        if osymbol_table is None: osymbol_table = isymbol_table
        cls.word_to_ilabel_map = isymbol_table.word_to_id_map
        cls.word_to_olabel_map = osymbol_table.word_to_id_map
        cls.olabel_to_word_map = osymbol_table.id_to_word_map
        cls.eps_like_ilabels = tuple(cls.word_to_ilabel_map[word] for word in cls.eps_like_words)
        cls.silent_olabels = tuple(
            frozenset(cls.word_to_olabel_map[word] for word in cls.silent_words)
            | frozenset(symbol for (word, symbol) in cls.word_to_olabel_map.items() if word.startswith('#nonterm')))
        cls.wildcard_nonterms = frozenset(wildcard_nonterms)
        cls.wildcard_olabels = tuple(cls.word_to_olabel_map[word] for word in cls.wildcard_nonterms)
        assert cls.word_to_ilabel_map[cls.eps] == 0

        cls.init_ffi()
        result = cls._lib.fst__init(len(cls.eps_like_ilabels), cls.eps_like_ilabels,
            len(cls.silent_olabels), cls.silent_olabels,
            len(cls.wildcard_olabels), cls.wildcard_olabels)
        if not result:
            raise WenetError("Failed fst__init")

    def __init__(self):
        super().__init__()
        self._construct()

    def _construct(self):
        self.native_obj = self._lib.fst__construct()
        if self.native_obj == _ffi.NULL:
            raise WenetError("Failed fst__construct")

        self.num_states = 1  # Is initialized with a start state
        self.num_arcs = 0
        self.filename = None
        self._compiled_native_obj = None

    def __del__(self):
        self.destruct()

    def destruct(self):
        del self.compiled_native_obj
        if self.native_obj is not None:
            result = self._lib.fst__destruct(self.native_obj)
            self.native_obj = None
            if not result:
                raise WenetError("Failed fst__destruct on %r" % self.native_obj)

    compiled_native_obj = property(lambda self: self._compiled_native_obj)
    @compiled_native_obj.setter
    def compiled_native_obj(self, value):
        del self.compiled_native_obj
        self._compiled_native_obj = value
    @compiled_native_obj.deleter
    def compiled_native_obj(self):
        if self._compiled_native_obj is not None:
            result = self._lib.fst__destruct(self._compiled_native_obj)
            self._compiled_native_obj = None
            if not result:
                raise WenetError("Failed fst__destruct on %r" % self._compiled_native_obj)

    def clear(self):
        self.destruct()
        self._construct()

    def add_state(self, weight=None, initial=False, final=False):
        """ Default weight is 1. """
        self.filename = None
        if weight is None:
            weight = 1 if final else 0
        else:
            assert final
        weight = -math.log(weight) if weight != 0 else self.zero
        id = self._lib.fst__add_state(self.native_obj, float(weight), bool(initial))
        if id < 0:
            raise WenetError("Failed fst__add_state")
        self.num_states += 1
        if initial:
            self.num_arcs += 1
        return id

    def add_arc(self, src_state, dst_state, label, olabel=None, weight=None):
        """ Default weight is 1. None label is replaced by eps. Default olabel of None is replaced by label. """
        self.filename = None
        if olabel is None: olabel = label
        if weight is None: weight = 1
        weight = -math.log(weight) if weight != 0 else self.zero
        label_id = self.word_to_ilabel_map[label] if label is not None else 0
        olabel_id = self.word_to_olabel_map[olabel] if olabel is not None else 0
        result = self._lib.fst__add_arc(self.native_obj, int(src_state), int(dst_state), int(label_id), int(olabel_id), float(weight))
        if not result:
            raise WenetError("Failed fst__add_arc")
        self.num_arcs += 1

    def compute_hash(self, dependencies_seed_hash_str='0'*32):
        hash_p = _ffi.new('char[]', 33)  # Length of MD5 hex string + null terminator
        result = self._lib.fst__compute_md5(self.native_obj, hash_p, encode(dependencies_seed_hash_str))
        if not result:
            raise WenetError("Failed fst__compute_md5")
        hash_str = decode(_ffi.string(hash_p))
        self.filename = hash_str + '.fst'
        return hash_str

    ####################################################################################################################

    def has_path(self):
        """ Returns True iff there is a path (from start state to a final state). Uses BFS. Assumes can nonterminals succeed. """
        result = self._lib.fst__has_path(self.native_obj)
        return result

    def has_eps_path(self, path_src_state, path_dst_state, eps_like_labels=frozenset()):
        """ Returns True iff there is a epsilon-like-only path from src_state to dst_state. Uses BFS. Does not follow nonterminals! """
        assert not eps_like_labels
        result = self._lib.fst__has_eps_path(self.native_obj, path_src_state, path_dst_state)
        return result

    def does_match(self, target_words, wildcard_nonterms=(), include_silent=False, output_max_length=1024):
        """ Returns the olabels on a matching path if there is one, False if not. Uses BFS. Wildcard accepts zero or more words. """
        # FIXME: do in decoder!
        assert frozenset(wildcard_nonterms) == self.wildcard_nonterms
        output_p = _ffi.new('int32_t[]', output_max_length)
        output_len_p = _ffi.new('int32_t*', output_max_length)
        target_labels = [self.word_to_ilabel_map[word] for word in target_words]
        result = self._lib.fst__does_match(self.native_obj, len(target_labels), target_labels, output_p, output_len_p)
        if output_len_p[0] > output_max_length:
            raise WenetError("fst__does_match needed too much output length")
        if result:
            return tuple(self.olabel_to_word_map[symbol]
                for symbol in output_p[0:output_len_p[0]]
                if include_silent or symbol not in self.silent_olabels)
        return False

    ####################################################################################################################

    def write_file(self, fst_filename):
        result = self._lib.fst__write_file(self.native_obj, encode(fst_filename))
        if not result:
            raise WenetError("Failed fst__write_file")

    def write_file_const(self, fst_filename):
        result = self._lib.fst__write_file_const(self.native_obj, encode(fst_filename))
        if not result:
            raise WenetError("Failed fst__write_file")

    def print(self, fst_filename=None):
        result = self._lib.fst__print(self.native_obj, (encode(fst_filename) if fst_filename is not None else _ffi.NULL))
        if not result:
            raise WenetError("Failed fst__print")

    @classmethod
    def load_file(cls, fst_filename):
        cls.init_ffi()
        native_obj = cls._lib.fst__load_file(encode(fst_filename))
        if not native_obj:
            raise WenetError("Failed fst__load_file")
        # FIXME: memory leak possible?
        return native_obj

    @classmethod
    def compile_text(cls, fst_text, isymbols_filename, osymbols_filename):
        cls.init_ffi()
        native_obj = cls._lib.fst__compile_text(encode(fst_text), encode(isymbols_filename), encode(osymbols_filename))
        if not native_obj:
            raise WenetError("Failed fst__compile_text")
        # FIXME: memory leak possible?
        return native_obj


########################################################################################################################

class SymbolTable(object):

    def __init__(self, filename=None):
        self.word_to_id_map = dict()
        self.id_to_word_map = dict()
        self.max_term_word_id = -1
        self.expand_word_to_id_map_func = None
        if filename is not None:
            self.load_text_file(filename)

    def load_text_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            word_id_pairs = [line.strip().split() for line in file]
        self.word_to_id_map.clear()
        self.id_to_word_map.clear()
        self.word_to_id_map.update({ word: int(id) for (word, id) in word_id_pairs })
        self.id_to_word_map.update({ id: word for (word, id) in self.word_to_id_map.items() })
        self.max_term_word_id = max(id for (word, id) in self.word_to_id_map.items() if not word.lower().startswith('#nonterm'))

    def expand_word_to_id_map(self, func):
        self.expand_word_to_id_map_func = func
        self.word_to_id_map.update({ func(word): id for (word, id) in self.word_to_id_map.items() })

    def add_word(self, word, id=None):
        if id is None:
            self.max_term_word_id += 1
            id = self.max_term_word_id
        else:
            id = int(id)
        self.word_to_id_map[word] = id
        self.id_to_word_map[id] = word
        if self.expand_word_to_id_map_func:
            self.word_to_id_map[self.expand_word_to_id_map_func(word)] = id

    words = property(lambda self: self.word_to_id_map.keys())

    def __contains__(self, word):
        return (word in self.word_to_id_map)
