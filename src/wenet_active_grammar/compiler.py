#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import collections, multiprocessing, os, re, threading
import concurrent.futures
from contextlib import contextmanager

from . import _log, WenetError
from .utils import debug_timer, show_donation_message
from .wfst import NativeWFST
from .model import Model
from .wrapper import WenetSTTModel, WenetAGDecoder

_log = _log.getChild('compiler')


########################################################################################################################

class WenetRule(object):

    cls_lock = threading.Lock()

    def __init__(self, compiler, name, nonterm=True, has_dictation=None, is_complex=None):
        """
        :param nonterm: bool whether rule represents a nonterminal in the active-grammar-fst (only False for the top FST?)
        """
        self.compiler = compiler
        self.name = name
        self.nonterm = nonterm
        self.has_dictation = has_dictation
        self.is_complex = is_complex

        # id: matches "nonterm:rule__"; 0-based; can/will change due to rule unloading!
        self.id = int(self.compiler.alloc_rule_id() if nonterm else -1)
        if self.id > self.compiler._max_rule_id: raise WenetError("WenetRule id > compiler._max_rule_id")
        if self.id in self.compiler.wenet_rule_by_id_dict: raise WenetError("WenetRule id already in use")
        if self.id >= 0:
            self.compiler.wenet_rule_by_id_dict[self.id] = self

        # Private/protected
        self._fst_text = None
        self.compiled = False
        self.loaded = False
        self.reloading = False  # WenetRule is in the process of the reload contextmanager
        self.has_been_loaded = False  # WenetRule was loaded, then reload() was called & completed, and now it is not currently loaded, and load() we need to call the decoder's reload
        self.destroyed = False  # WenetRule must not be used/referenced anymore

        # Public
        self.fst = NativeWFST()
        self.matcher = None
        self.active = True

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.id, self.name)

    fst_cache = property(lambda self: self.compiler.fst_cache)
    decoder = property(lambda self: self.compiler.decoder)

    pending_compile = property(lambda self: (self in self.compiler.compile_queue) or (self in self.compiler.compile_duplicate_filename_queue))
    pending_load = property(lambda self: self in self.compiler.load_queue)

    fst_wrapper = property(lambda self: self.fst if self.fst.native else self.filepath)
    filename = property(lambda self: self.fst.filename)

    @property
    def filepath(self):
        assert self.filename
        assert self.compiler.tmp_dir is not None
        return os.path.join(self.compiler.tmp_dir, self.filename)

    def compile(self, lazy=False, duplicate=None):
        if self.destroyed: raise WenetError("Cannot use a WenetRule after calling destroy()")
        if self.compiled: return self

        if self.fst.native:
            if not self.filename:
                self.fst.compute_hash(self.fst_cache.dependencies_hash)
                assert self.filename

        else:
            # Handle compiling text WFST to binary
            if not self._fst_text:
                # self.fst.normalize_weights()
                self._fst_text = self.fst.get_fst_text(fst_cache=self.fst_cache)
                assert self.filename
            # if 'dictation' in self._fst_text: _log.log(50, '\n    '.join(["%s: FST text:" % self] + self._fst_text.splitlines()))  # log _fst_text

        if self.compiler.cache_fsts and self.fst_cache.fst_is_current(self.filepath, touch=True):
            _log.debug("%s: Skipped FST compilation thanks to FileCache" % self)
            if self.compiler.decoding_framework == 'agf' and self.fst.native:
                self.fst.compiled_native_obj = NativeWFST.load_file(self.filepath)
            self.compiled = True
            return self
        else:
            if duplicate:
                _log.warning("%s was supposed to be a duplicate compile, but was not found in FileCache")

        if lazy:
            if not self.pending_compile:
                # Special handling for rules that are an exact content match (and hence hash/name) with another (different) rule already in the compile_queue
                if not any(self.filename == wenet_rule.filename for wenet_rule in self.compiler.compile_queue if self != wenet_rule):
                    self.compiler.compile_queue.add(self)
                else:
                    self.compiler.compile_duplicate_filename_queue.add(self)
            return self

        return self.finish_compile()

    def finish_compile(self):
        # Must be thread-safe!
        with self.cls_lock:
            self.compiler.prepare_for_compilation()
        _log.log(15, "%s: Compiling %sstate/%sarc FST%s%s" % (self, self.fst.num_states, self.fst.num_arcs,
                (" (%dbyte)" % len(self._fst_text)) if self._fst_text else "",
                (" to " + self.filename) if self.filename else ""))
        assert self.fst.native or self._fst_text
        if _log.isEnabledFor(3):
            if self.fst.native: self.fst.write_file('tmp_G.fst')
            if _log.isEnabledFor(2):
                if self._fst_text: _log.log(2, '\n    '.join(["%s: FST text:" % self] + self._fst_text.splitlines()))  # log _fst_text
                elif self.fst.native: self.fst.print()

        try:
            if self.compiler.decoding_framework == 'agf':
                if self.fst.native:
                    self.fst.compiled_native_obj = self.compiler._compile_agf_graph(compile=True, nonterm=self.nonterm, input_fst=self.fst, return_output_fst=True,
                        output_filename=(self.filepath if self.compiler.cache_fsts else None))
                else:
                    self.compiler._compile_agf_graph(compile=True, nonterm=self.nonterm, input_text=self._fst_text, output_filename=self.filepath)
                    self._fst_text = None  # Free memory

            elif self.compiler.decoding_framework == 'laf':
                # self.compiler._compile_laf_graph(compile=True, nonterm=self.nonterm, input_text=self._fst_text, output_filename=self.filepath)
                # Keep self._fst_text, for adding directly later
                pass

            else: raise WenetError("unknown compiler.decoding_framework")
        except Exception as e:
            raise WenetError("Exception while compiling", self)  # Return this WenetRule inside exception

        self.compiled = True
        return self

    def load(self, lazy=False):
        if self.destroyed: raise WenetError("Cannot use a WenetRule after calling destroy()")
        if lazy or self.pending_compile:
            self.compiler.load_queue.add(self)
            return self
        assert self.compiled

        if self.has_been_loaded:
            # FIXME: why is this necessary?
            self._do_reloading()
        else:
            if self.compiler.decoding_framework == 'agf':
                grammar_fst_index = self.decoder.add_grammar_fst(self.fst if self.fst.native else self.filepath)
            elif self.compiler.decoding_framework == 'laf':
                grammar_fst_index = self.decoder.add_grammar_fst(self.fst) if self.fst.native else self.decoder.add_grammar_fst_text(self._fst_text)
            else: raise WenetError("unknown compiler decoding_framework")
            assert self.id == grammar_fst_index, "add_grammar_fst allocated invalid grammar_fst_index %d != %d for %s" % (grammar_fst_index, self.id, self)

        self.loaded = True
        self.has_been_loaded = True
        return self

    def _do_reloading(self):
        if self.compiler.decoding_framework == 'agf':
            return self.decoder.reload_grammar_fst(self.id, (self.fst if self.fst.native else self.filepath))
        elif self.compiler.decoding_framework == 'laf':
            assert self.fst.native
            return self.decoder.reload_grammar_fst(self.id, self.fst)
            if not self.fst.native: return self.decoder.reload_grammar_fst_text(self.id, self._fst_text)  # FIXME: not implemented
        else: raise WenetError("unknown compiler decoding_framework")

    @contextmanager
    def reload(self):
        """ Used for modifying a rule in place, e.g. ListRef. """
        if self.destroyed: raise WenetError("Cannot use a WenetRule after calling destroy()")

        was_loaded = self.loaded
        self.reloading = True
        self.fst.clear()
        self._fst_text = None
        self.compiled = False
        self.loaded = False

        yield

        if self.compiled and was_loaded:
            if not self.loaded:
                # FIXME: how is this different from the branch of the if above in load()?
                self._do_reloading()
                self.loaded = True
        elif was_loaded:  # must be not self.compiled (i.e. the compile during reloading was lazy)
            self.compiler.load_queue.add(self)
        self.reloading = False

    def destroy(self):
        """ Destructor. Unloads rule. The rule should not be used/referenced anymore after calling! """
        if self.destroyed:
            return

        if self.loaded:
            self.decoder.remove_grammar_fst(self.id)
            assert self not in self.compiler.compile_queue
            assert self not in self.compiler.compile_duplicate_filename_queue
            assert self not in self.compiler.load_queue
        else:
            if self in self.compiler.compile_queue: self.compiler.compile_queue.remove(self)
            if self in self.compiler.compile_duplicate_filename_queue: self.compiler.compile_duplicate_filename_queue.remove(self)
            if self in self.compiler.load_queue: self.compiler.load_queue.remove(self)

        # Adjust other wenet_rules ids down, if above self.id, then rebuild dict
        other_wenet_rules = list(self.compiler.wenet_rule_by_id_dict.values())
        other_wenet_rules.remove(self)
        for wenet_rule in other_wenet_rules:
            if wenet_rule.id > self.id:
                wenet_rule.id -= 1
        self.compiler.wenet_rule_by_id_dict = { wenet_rule.id: wenet_rule for wenet_rule in other_wenet_rules }

        self.compiler.free_rule_id()
        self.destroyed = True


########################################################################################################################

class Compiler(object):

    def __init__(self, model_dir, alternative_dictation=None):

        show_donation_message()
        self._log = _log

        self.parsing_framework = 'token'
        assert self.parsing_framework in ('token', 'text')
        self.alternative_dictation = alternative_dictation

        self.model = Model(model_dir)
        self._lexicon_files_stale = False

        self.model.words_table.expand_word_to_id_map(lambda word: word.lower())
        NativeWFST.init_class(osymbol_table=self.model.words_table, isymbol_table=self.model.words_table, wildcard_nonterms=tuple())
        # FIXME: self.wildcard_nonterms
        self.decoder = None

        self._num_wenet_rules = 0
        self._max_rule_id = 999
        self.nonterminals = tuple(['#nonterm:dictation'] + ['#nonterm:rule%i' % i for i in range(self._max_rule_id + 1)])
        words_set = frozenset(self.model.words_table.words)
        self._oov_word = '<unk>' if ('<unk>' in self.model.words_table) else None  # FIXME: make this configurable, for different models
        self._silence_words = frozenset(['!SIL']) & words_set  # FIXME: make this configurable, for different models
        self._noise_words = frozenset(['<unk>', '!SIL']) & words_set  # FIXME: make this configurable, for different models

        self.wenet_rule_by_id_dict = collections.OrderedDict()  # maps WenetRule.id -> WenetRule
        self.compile_queue = set()  # WenetRule
        self.load_queue = set()  # WenetRule; must maintain same order as order of instantiation!

    def init_decoder(self, config=None):
        if self.decoder: raise WenetError("Decoder already initialized")
        # if dictation_fst_file is None: dictation_fst_file = self.dictation_fst_filepath
        config = dict(
            max_num_rules=self._max_rule_id+1,
            grammar_symbol_path=self.model.files_dict['words.txt'],
            rule0_label='#nonterm:rule0',
            **({} if config is None else config),
        )
        self.decoder = WenetAGDecoder(WenetSTTModel(WenetSTTModel.build_config(self.model.model_dir, config)))
        return self.decoder

    num_wenet_rules = property(lambda self: self._num_wenet_rules)
    lexicon_words = property(lambda self: self.model.words_table.word_to_id_map)
    _longest_word = property(lambda self: self.model.longest_word)

    def alloc_rule_id(self):
        id = self._num_wenet_rules
        self._num_wenet_rules += 1
        return id

    def free_rule_id(self):
        id = self._num_wenet_rules
        self._num_wenet_rules -= 1
        return id


    ####################################################################################################################
    # Methods for compiling graphs.

    def add_word(self, word, phones=None, lazy_compilation=False, allow_online_pronunciations=False):
        pronunciations = self.model.add_word(word, phones=phones, lazy_compilation=lazy_compilation, allow_online_pronunciations=allow_online_pronunciations)
        self._lexicon_files_stale = True  # Only mark lexicon stale if it was successfully modified (not an exception)
        return pronunciations

    def prepare_for_compilation(self):
        if self._lexicon_files_stale:
            self.model.generate_lexicon_files()
            self.model.load_words()  # FIXME: This re-loading from the words.txt file may be unnecessary now that we have/use NativeWFST + SymbolTable, but it's not clear if it's safe to remove it.
            self.decoder.load_lexicon()
            if self._agf_compiler:
                # TODO: Just update the necessary files in the config
                self._agf_compiler.destroy()
                self._agf_compiler = self._init_agf_compiler()
            self._lexicon_files_stale = False

    def compile_top_fst(self):
        return self._build_top_fst(nonterms=['#nonterm:rule'+str(i) for i in range(self._max_rule_id + 1)], noise_words=self._noise_words).compile()

    def compile_top_fst_dictation_only(self):
        return self._build_top_fst(nonterms=['#nonterm:dictation'], noise_words=self._noise_words).compile()

    def _build_top_fst(self, nonterms, noise_words):
        wenet_rule = WenetRule(self, 'top', nonterm=False)
        fst = wenet_rule.fst
        state_initial = fst.add_state(initial=True)
        state_final = fst.add_state(final=True)

        state_return = fst.add_state()
        for nonterm in nonterms:
            fst.add_arc(state_initial, state_return, nonterm)
        fst.add_arc(state_return, state_final, None, '#nonterm:end')

        if noise_words:
            for (state_from, state_to) in [
                    (state_initial, state_final),
                    # (state_initial, state_initial),  # FIXME: test this
                    # (state_final, state_final),
                    ]:
                for word in noise_words:
                    fst.add_arc(state_from, state_to, word)

        return wenet_rule

    def process_compile_and_load_queues(self):
        # Clean out obsolete entries
        self.compile_queue.difference_update([wenet_rule for wenet_rule in self.compile_queue if wenet_rule.compiled])
        self.compile_duplicate_filename_queue.difference_update([wenet_rule for wenet_rule in self.compile_duplicate_filename_queue if wenet_rule.compiled])
        self.load_queue.difference_update([wenet_rule for wenet_rule in self.load_queue if wenet_rule.loaded])

        if self.compile_queue or self.compile_duplicate_filename_queue or self.load_queue:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = executor.map(lambda wenet_rule: wenet_rule.finish_compile(), self.compile_queue)
                # Load pending rules that have already been compiled
                # for wenet_rule in (self.load_queue - self.compile_queue - self.compile_duplicate_filename_queue):
                #     wenet_rule.load()
                #     self.load_queue.remove(wenet_rule)
                # Handle rules as they are completed (have been compiled)
                for wenet_rule in results:
                    assert wenet_rule.compiled
                    self.compile_queue.remove(wenet_rule)
                    # if wenet_rule in self.load_queue:
                    #     wenet_rule.load()
                    #     self.load_queue.remove(wenet_rule)
                # Handle rules that were pending compile but were duplicate and so compiled by/for another rule. They should be in the cache now
                for wenet_rule in list(self.compile_duplicate_filename_queue):
                    wenet_rule.compile(duplicate=True)
                    assert wenet_rule.compiled
                    self.compile_duplicate_filename_queue.remove(wenet_rule)
                    # if wenet_rule in self.load_queue:
                    #     wenet_rule.load()
                    #     self.load_queue.remove(wenet_rule)
                # Load rules in correct order
                for wenet_rule in sorted(self.load_queue, key=lambda kr: kr.id):
                    wenet_rule.load()
                    assert wenet_rule.loaded
                    self.load_queue.remove(wenet_rule)


    ####################################################################################################################
    # Methods for recognition.

    def prepare_for_recognition(self):
        try:
            if self.compile_queue or self.compile_duplicate_filename_queue or self.load_queue:
                self.process_compile_and_load_queues()
        except WenetError:
            raise
        except Exception:
            raise WenetError("Exception while compiling/loading rules in prepare_for_recognition")
        finally:
            if self.fst_cache.dirty:
                self.fst_cache.save()

    wildcard_nonterms = ('#nonterm:dictation', '#nonterm:dictation_cloud')

    def parse_output_for_rule(self, wenet_rule, output):
        """Can be used even when self.parsing_framework == 'token', only for mimic (which contains no nonterms)."""
        labels = wenet_rule.fst.does_match(output.split(), wildcard_nonterms=self.wildcard_nonterms)
        self._log.log(5, "parse_output_for_rule(%s, %r) got %r", wenet_rule, output, labels)
        if labels is False:
            return None
        words = [label for label in labels if not label.startswith('#nonterm:')]
        parsed_output = ' '.join(words)
        if parsed_output.lower() != output:
            self._log.error("parsed_output(%r).lower() != output(%r)" % (parsed_output, output))
        return words

    alternative_dictation_regex = re.compile(r'(?<=#nonterm:dictation_cloud )(.*?)(?= #nonterm:end)')  # lookbehind & lookahead assertions

    def parse_output(self, output, dictation_info_func=None):
        assert self.parsing_framework == 'token'
        self._log.debug("parse_output(%r)" % output)
        if (output == '') or (output in self._noise_words):
            return None, [], []

        nonterm_token, _, parsed_output = output.partition(' ')
        assert nonterm_token.startswith('#nonterm:rule')
        wenet_rule_id = int(nonterm_token[len('#nonterm:rule'):])
        wenet_rule = self.wenet_rule_by_id_dict[wenet_rule_id]

        if self.alternative_dictation and dictation_info_func and wenet_rule.has_dictation and '#nonterm:dictation_cloud' in parsed_output:
            try:
                if callable(self.alternative_dictation):
                    alternative_text_func = self.alternative_dictation
                else:
                    raise TypeError("Invalid alternative_dictation value: %r" % self.alternative_dictation)

                audio_data, word_align = dictation_info_func()
                self._log.log(5, "alternative_dictation word_align: %s", word_align)
                words, times, lengths = list(zip(*word_align))
                # Find start & end word-index & byte-offset of each alternative dictation span
                dictation_spans = [{
                        'index_start': index,
                        'offset_start': time,
                        'index_end': words.index('#nonterm:end', index),
                        'offset_end': times[words.index('#nonterm:end', index)],
                    }
                    for index, (word, time, length) in enumerate(word_align)
                    if word.startswith('#nonterm:dictation_cloud')]

                # If last dictation is at end of utterance, include rest of audio_data; else, include half of audio_data between dictation end and start of next word
                dictation_span = dictation_spans[-1]
                if dictation_span['index_end'] == len(word_align) - 1:
                    dictation_span['offset_end'] = len(audio_data)
                else:
                    next_word_time = times[dictation_span['index_end'] + 1]
                    dictation_span['offset_end'] = (dictation_span['offset_end'] + next_word_time) // 2

                def replace_dictation(matchobj):
                    orig_text = matchobj.group(1)
                    dictation_span = dictation_spans.pop(0)
                    dictation_audio = audio_data[dictation_span['offset_start'] : dictation_span['offset_end']]
                    kwargs = dict(language_code=self.cloud_dictation_lang)
                    with debug_timer(self._log.debug, 'alternative_dictation call'):
                        alternative_text = alternative_text_func(dictation_audio, **kwargs)
                        self._log.debug("alternative_dictation: %.2fs audio -> %r", (0.5 * len(dictation_audio) / 16000), alternative_text)  # FIXME: hardcoded sample_rate!
                    # alternative_dictation.write_wav('test.wav', dictation_audio)
                    return (alternative_text or orig_text)

                parsed_output = self.alternative_dictation_regex.sub(replace_dictation, parsed_output)
            except Exception as e:
                self._log.exception("Exception performing alternative dictation")

        words = []
        words_are_dictation_mask = []
        in_dictation = False
        for word in parsed_output.split():
            if word.startswith('#nonterm:'):
                if word.startswith('#nonterm:dictation'):
                    in_dictation = True
                elif in_dictation and word == '#nonterm:end':
                    in_dictation = False
            else:
                words.append(word)
                words_are_dictation_mask.append(in_dictation)

        return wenet_rule, words, words_are_dictation_mask

    def parse_partial_output(self, output):
        assert self.parsing_framework == 'token'
        self._log.log(3, "parse_partial_output(%r)", output)
        if (output == '') or (output in self._noise_words):
            return None, [], [], False

        nonterm_token, _, parsed_output = output.partition(' ')
        assert nonterm_token.startswith('#nonterm:rule')
        wenet_rule_id = int(nonterm_token[len('#nonterm:rule'):])
        wenet_rule = self.wenet_rule_by_id_dict[wenet_rule_id]

        words = []
        words_are_dictation_mask = []
        in_dictation = False
        for word in parsed_output.split():
            if word.startswith('#nonterm:'):
                if word.startswith('#nonterm:dictation'):
                    in_dictation = True
                elif in_dictation and word == '#nonterm:end':
                    in_dictation = False
            else:
                words.append(word)
                words_are_dictation_mask.append(in_dictation)

        return wenet_rule, words, words_are_dictation_mask, in_dictation

########################################################################################################################
# Utility functions.

def remove_words_in_words(words, remove_words_func):
    return [word for word in words if not remove_words_func(word)]

def remove_words_in_text(text, remove_words_func):
    return ' '.join(word for word in text.split() if not remove_words_func(word))

def remove_nonterms_in_words(words):
    return remove_words_in_words(words, lambda word: word.startswith('#nonterm:'))

def remove_nonterms_in_text(text):
    return remove_words_in_text(text, lambda word: word.startswith('#nonterm:'))
