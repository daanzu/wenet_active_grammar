#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import os, re, shutil
from io import open

from six import PY2, text_type

from . import _log, WenetError
from .wfst import SymbolTable
from .utils import find_file, load_symbol_table, show_donation_message
import wenet_active_grammar.utils as utils

_log = _log.getChild('model')


########################################################################################################################

class Lexicon(object):

    def __init__(self, phones):
        """ phones: list of strings, each being a phone """
        self.phone_set = set(self.make_position_independent(phones))

    # XSAMPA phones are 1-letter each, so 2-letter below represent 2 separate phones.
    CMU_to_XSAMPA_dict = {
        "'"   : "'",
        'AA'  : 'A',
        'AE'  : '{',
        'AH'  : 'V',  ##
        'AO'  : 'O',  ##
        'AW'  : 'aU',
        'AY'  : 'aI',
        'B'   : 'b',
        'CH'  : 'tS',
        'D'   : 'd',
        'DH'  : 'D',
        'EH'  : 'E',
        'ER'  : '3',
        'EY'  : 'eI',
        'F'   : 'f',
        'G'   : 'g',
        'HH'  : 'h',
        'IH'  : 'I',
        'IY'  : 'i',
        'JH'  : 'dZ',
        'K'   : 'k',
        'L'   : 'l',
        'M'   : 'm',
        'NG'  : 'N',
        'N'   : 'n',
        'OW'  : 'oU',
        'OY'  : 'OI', ##
        'P'   : 'p',
        'R'   : 'r',
        'SH'  : 'S',
        'S'   : 's',
        'TH'  : 'T',
        'T'   : 't',
        'UH'  : 'U',
        'UW'  : 'u',
        'V'   : 'v',
        'W'   : 'w',
        'Y'   : 'j',
        'ZH'  : 'Z',
        'Z'   : 'z',
    }
    CMU_to_XSAMPA_dict.update({'AX': '@'})
    del CMU_to_XSAMPA_dict["'"]
    XSAMPA_to_CMU_dict = { v: k for k,v in CMU_to_XSAMPA_dict.items() }  # FIXME: handle double-entries

    @classmethod
    def phones_cmu_to_xsampa_generic(cls, phones, lexicon_phones=None):
        new_phones = []
        for phone in phones:
            stress = False
            if phone.endswith('1'):
                phone = phone[:-1]
                stress = True
            elif phone.endswith(('0', '2')):
                phone = phone[:-1]
            phone = cls.CMU_to_XSAMPA_dict[phone]
            assert 1 <= len(phone) <= 2

            new_phone = ("'" if stress else '') + phone
            if (lexicon_phones is not None) and (new_phone in lexicon_phones):
                # Add entire possibly-2-letter phone
                new_phones.append(new_phone)
            else:
                # Add each individual 1-letter phone
                for match in re.finditer(r"('?).", new_phone):
                    new_phones.append(match.group(0))

        return new_phones

    def phones_cmu_to_xsampa(self, phones):
        return self.phones_cmu_to_xsampa_generic(phones, self.phone_set)

    @classmethod
    def make_position_dependent(cls, phones):
        if len(phones) == 0: return []
        elif len(phones) == 1: return [phones[0]+'_S']
        else: return [phones[0]+'_B'] + [phone+'_I' for phone in phones[1:-1]] + [phones[-1]+'_E']

    @classmethod
    def make_position_independent(cls, phones):
        return [re.sub(r'_[SBIE]', '', phone) for phone in phones]

    @classmethod
    def generate_pronunciations_cmu_online(cls, word):
        try:
            import requests
            files = {'wordfile': ('wordfile', word)}
            req = requests.post('http://www.speech.cs.cmu.edu/cgi-bin/tools/logios/lextool.pl', files=files)
            req.raise_for_status()
            # FIXME: handle network failures
            match = re.search(r'<!-- DICT (.*)  -->', req.text)
            if match:
                url = match.group(1)
                req = requests.get(url)
                req.raise_for_status()
                entries = req.text.strip().split('\n')
                pronunciations = []
                for entry in entries:
                    tokens = entry.strip().split()
                    assert re.match(word + r'(\(\d\))?', tokens[0], re.I)  # 'SEMI-COLON' or 'SEMI-COLON(2)'
                    phones = tokens[1:]
                    _log.debug("generated pronunciation with cloud-cmudict for %r: CMU phones are %r" % (word, phones))
                    pronunciations.append(phones)
                return pronunciations
            raise WenetError("received bad response from www.speech.cs.cmu.edu: %r" % req.text)
        except Exception as e:
            _log.exception("generate_pronunciations exception accessing www.speech.cs.cmu.edu")
            raise e

    g2p_en = None

    @classmethod
    def attempt_load_g2p_en(cls, model_dir=None):
        try:
            if model_dir:
                import nltk
                nltk.data.path.insert(0, os.path.abspath(os.path.join(model_dir, 'g2p')))
            # g2p_en>=2.1.0
            import g2p_en
            cls.g2p_en = g2p_en.G2p()
            assert all(re.sub(r'[012]$', '', phone) in cls.CMU_to_XSAMPA_dict for phone in cls.g2p_en.phonemes if not phone.startswith('<'))
        except Exception:  # including ImportError
            cls.g2p_en = False  # Don't try anymore.
            _log.debug("failed to load g2p_en")

    @classmethod
    def generate_pronunciations_g2p_en(cls, word):
        try:
            phones = cls.g2p_en(word)
            _log.debug("generated pronunciation with g2p_en for %r: %r" % (word, phones))
            return [phones]
        except Exception as e:
            _log.exception("generate_pronunciations exception using g2p_en")
            raise e

    @classmethod
    def generate_pronunciations(cls, word, model_dir=None, allow_online_pronunciations=False):
        """returns CMU/arpabet phones"""
        if cls.g2p_en is None:
            cls.attempt_load_g2p_en(model_dir)
        if cls.g2p_en:
            return cls.generate_pronunciations_g2p_en(word)
        if allow_online_pronunciations:
            return cls.generate_pronunciations_cmu_online(word)
        raise WenetError("cannot generate word pronunciation: no generators available")


########################################################################################################################

class Model(object):
    def __init__(self, model_dir):
        show_donation_message()

        self.model_dir = os.path.join(model_dir, '')
        if not os.path.isdir(self.model_dir):
            raise WenetError("cannot find model_dir: %r" % self.model_dir)

        # version_file = os.path.join(self.model_dir, 'KAG_VERSION')
        # if os.path.isfile(version_file):
        #     with open(version_file, 'r', encoding='utf-8') as f:
        #         model_version = f.read().strip()
        #         if model_version != REQUIRED_MODEL_VERSION:
        #             raise WenetError("invalid model_dir version! please download a compatible model")
        # else:
        #     _log.warning("model_dir has no version information; errors below may indicate an incompatible model")

        # self.create_missing_files()
        # self.check_user_lexicon()

        self.files_dict = {
            'model_dir': self.model_dir,
            'words.txt': find_file(self.model_dir, 'fst_words.txt', default=True),
            'units.txt': find_file(self.model_dir, 'words.txt', default=True),
            # 'words.base.txt': find_file(self.model_dir, 'words.base.txt', default=True),
            # 'user_lexicon.txt': find_file(self.model_dir, 'user_lexicon.txt', default=True),
            # 'nonterminals.txt': find_file(self.model_dir, 'nonterminals.txt', default=True),
        }
        self.files_dict.update({ k.replace('.', '_'): v for (k, v) in self.files_dict.items() })  # For named placeholder access in str.format()
        # self.fst_cache = utils.FSTFileCache(os.path.join(self.model_dir, defaults.FILE_CACHE_FILENAME), dependencies_dict=self.files_dict, tmp_dir=self.tmp_dir)

        # self.phone_to_int_dict = { phone: i for phone, i in load_symbol_table(self.files_dict['phones.txt']) }
        # self.lexicon = Lexicon(self.phone_to_int_dict.keys())
        # self.nonterm_phones_offset = self.phone_to_int_dict.get('#nonterm_bos')
        # if self.nonterm_phones_offset is None: raise WenetError("missing nonterms in 'phones.txt'")
        # self.nonterm_words_offset = symbol_table_lookup(self.files_dict['words.base.txt'], '#nonterm_begin')
        # if self.nonterm_words_offset is None: raise WenetError("missing nonterms in 'words.base.txt'")

        # Update files if needed, before loading words
        # necessary_files = ['user_lexicon.txt', 'words.txt',]
        # non_lazy_files = ['align_lexicon.int', 'lexiconp_disambig.txt', 'L_disambig.fst',]
        # files_are_not_current = lambda files: any(not self.fst_cache.file_is_current(self.files_dict[file]) for file in files)
        # if self.fst_cache.cache_is_new or files_are_not_current(necessary_files + non_lazy_files):
        #     self.generate_lexicon_files()

        self.words_table = SymbolTable()
        self.load_words()

    def load_words(self, words_file=None):
        if words_file is None: words_file = self.files_dict['words.txt']
        _log.debug("loading words from %r", words_file)
        invalid_words = "<eps> !SIL <UNK> #0 <s> </s>".lower().split()
        self.words_table.load_text_file(words_file)
        self.longest_word = max(self.words_table.word_to_id_map.keys(), key=len)
        return self.words_table

    def read_user_lexicon(self, filename=None):
        if filename is None: filename = self.files_dict['user_lexicon.txt']
        with open(filename, 'r', encoding='utf-8') as file:
            entries = [line.split() for line in file if line.split()]
            for tokens in entries:
                # word lowercase
                tokens[0] = tokens[0].lower()
        return entries

    def write_user_lexicon(self, entries, filename=None):
        if filename is None: filename = self.files_dict['user_lexicon.txt']
        lines = [' '.join(tokens) + '\n' for tokens in entries]
        with open(filename, 'w', encoding='utf-8', newline='\n') as file:
            file.writelines(lines)

    def add_word(self, word, phones=None, lazy_compilation=False, allow_online_pronunciations=False):
        word = word.strip().lower()

        if phones is None:
            # Not given pronunciation(s), so generate pronunciation(s), then call ourselves recursively for each individual pronunciation
            pronunciations = Lexicon.generate_pronunciations(word, model_dir=self.model_dir, allow_online_pronunciations=allow_online_pronunciations)
            pronunciations = sum([
                self.add_word(word, phones, lazy_compilation=True)
                for phones in pronunciations], [])
            if not lazy_compilation:
                self.generate_lexicon_files()
            return pronunciations
            # FIXME: refactor this function

        # Now just handle single-pronunciation case...
        phones = self.lexicon.phones_cmu_to_xsampa(phones)
        new_entry = [word] + phones

        entries = self.read_user_lexicon()
        if any(new_entry == entry for entry in entries):
            _log.warning("word & pronunciation already in user_lexicon")
            return [phones]
        for tokens in entries:
            if word == tokens[0]:
                _log.warning("word (with different pronunciation) already in user_lexicon: %s" % tokens[1:])

        entries.append(new_entry)
        self.write_user_lexicon(entries)

        if lazy_compilation:
            self.words_table.add_word(word)
        else:
            self.generate_lexicon_files()

        return [phones]

    def create_missing_files(self):
        utils.touch_file(os.path.join(self.model_dir, 'user_lexicon.txt'))
        def check_file(filename, src_filename):
            # Create missing file from its base file
            if not find_file(self.model_dir, filename):
                src = find_file(self.model_dir, src_filename)
                dst = src.replace(src_filename, filename)
                shutil.copyfile(src, dst)
        check_file('words.txt', 'words.base.txt')
        check_file('align_lexicon.int', 'align_lexicon.base.int')
        check_file('lexiconp_disambig.txt', 'lexiconp_disambig.base.txt')

    def check_user_lexicon(self):
        """ Checks for a user lexicon file in the CWD, and if found and different than the model's user lexicon, extends the model's. """
        cwd_user_lexicon_filename = os.path.abspath('user_lexicon.txt')
        model_user_lexicon_filename = os.path.abspath(os.path.join(self.model_dir, 'user_lexicon.txt'))
        if (cwd_user_lexicon_filename != model_user_lexicon_filename) and os.path.isfile(cwd_user_lexicon_filename):
            cwd_user_lexicon_entries = [tuple(tokens) for tokens in self.read_user_lexicon(filename=cwd_user_lexicon_filename)]
            model_user_lexicon_entries = [tuple(tokens) for tokens in self.read_user_lexicon(filename=model_user_lexicon_filename)]
            model_user_lexicon_entries_set = set(model_user_lexicon_entries)
            new_user_lexicon_entries = [tokens for tokens in cwd_user_lexicon_entries if tokens not in model_user_lexicon_entries_set]
            if new_user_lexicon_entries:
                _log.info("adding new user lexicon entries from %r", cwd_user_lexicon_filename)
                entries = model_user_lexicon_entries + new_user_lexicon_entries
                self.write_user_lexicon(entries, filename=model_user_lexicon_filename)

    def generate_lexicon_files(self):
        """ Generates: words.txt, align_lexicon.int, lexiconp_disambig.txt, L_disambig.fst """
        _log.info("generating lexicon files")
        self.fst_cache.invalidate()

        # FIXME: refactor this to use words_table/SymbolTable
        max_word_id = max(word_id for word, word_id in load_symbol_table(base_filepath(self.files_dict['words.txt'])) if word_id < self.nonterm_words_offset)

        user_lexicon_entries = []
        with open(self.files_dict['user_lexicon.txt'], 'r', encoding='utf-8') as user_lexicon:
            for line in user_lexicon:
                tokens = line.split()
                if len(tokens) >= 2:
                    word, phones = tokens[0], tokens[1:]
                    phones = Lexicon.make_position_dependent(phones)
                    unknown_phones = [phone for phone in phones if phone not in self.phone_to_int_dict]
                    if unknown_phones:
                        raise WenetError("word %r has unknown phone(s) %r" % (word, unknown_phones))
                        # _log.critical("word %r has unknown phone(s) %r so using junk phones!!!", word, unknown_phones)
                        # phones = [phone if phone not in self.phone_to_int_dict else self.noise_phone for phone in phones]
                        # continue
                    max_word_id += 1
                    user_lexicon_entries.append((word, max_word_id, phones))

        def generate_file_from_base_with_user_lexicon(filename, write_func):
            filepath = self.files_dict[filename]
            with open(base_filepath(filepath), 'r', encoding='utf-8') as file:
                base_data = file.read()
            with open(filepath, 'w', encoding='utf-8', newline='\n') as file:
                file.write(base_data)
                for word, word_id, phones in user_lexicon_entries:
                    file.write(write_func(word, word_id, phones) + '\n')

        generate_file_from_base_with_user_lexicon('words.txt', lambda word, word_id, phones:
            str_space_join([word, word_id]))
        generate_file_from_base_with_user_lexicon('align_lexicon.int', lambda word, word_id, phones:
            str_space_join([word_id, word_id] + [self.phone_to_int_dict[phone] for phone in phones]))
        generate_file_from_base_with_user_lexicon('lexiconp_disambig.txt', lambda word, word_id, phones:
            '%s\t1.0 %s' % (word, ' '.join(phones)))

        lexicon_fst_text = WenetModelBuildUtils.make_lexicon_fst(
            left_context_phones=self.files_dict['left_context_phones_txt'],
            nonterminals=self.files_dict['nonterminals_txt'],
            sil_prob=0.5,
            sil_phone='SIL',
            sil_disambig='#14',  # FIXME: lookup correct value
            lexiconp=self.files_dict['lexiconp_disambig_txt'],
        )
        WenetModelBuildUtils.build_L_disambig(
            lexicon_fst_text.encode(encoding='latin-1'),
            phones_file=self.files_dict['phones_txt'], words_file=self.files_dict['words_txt'],
            wdisambig_phones_file=self.files_dict['wdisambig_phones_int'], wdisambig_words_file=self.files_dict['wdisambig_words_int'],
            fst_out_file=self.files_dict['L_disambig_fst'])

        # FIXME: generate_words_relabeled_file(self.files_dict['words.txt'], self.files_dict['relabel_ilabels.int'], self.files_dict['words.relabeled.txt'])

        self.fst_cache.update_dependencies()
        self.fst_cache.save()

    def reset_user_lexicon(self):
        utils.clear_file(self.files_dict['user_lexicon.txt'])
        self.generate_lexicon_files()

    @staticmethod
    def generate_words_relabeled_file(words_filename, relabel_filename, words_relabel_filename):
        """ generate a version of the words file, that has already been relabeled with the given relabel file """
        with open(words_filename, 'r', encoding='utf-8') as file:
            word_id_pairs = [(word, id) for (word, id) in [line.strip().split() for line in file]]
        with open(relabel_filename, 'r', encoding='utf-8') as file:
            relabel_map = {from_id: to_id for (from_id, to_id) in [line.strip().split() for line in file]}
        word_ids = frozenset(id for (word, id) in word_id_pairs)
        relabel_from_ids = frozenset(from_id for from_id in relabel_map.keys())
        if word_ids < relabel_from_ids:
            _log.warning("generate_words_relabeled_file: word_ids < relabel_from_ids")
        # if word_ids > relabel_from_ids:
        #     _log.warning("generate_words_relabeled_file: word_ids > relabel_from_ids")
        with open(words_relabel_filename, 'w', encoding='utf-8') as file:
            for (word, id) in word_id_pairs:
                file.write("%s %s\n" % (word, (relabel_map.get(id, id))))


########################################################################################################################

def convert_generic_model_to_agf(src_dir, model_dir):
    from .compiler import Compiler
    if PY2:
        from .wenet import augment_phones_txt_py2 as augment_phones_txt, augment_words_txt_py2 as augment_words_txt
    else:
        from .wenet import augment_phones_txt, augment_words_txt

    filenames = [
        'words.txt',
        'phones.txt',
        'align_lexicon.int',
        'disambig.int',
        # 'L_disambig.fst',
        'tree',
        'final.mdl',
        'lexiconp.txt',
        'word_boundary.txt',
        'optional_silence.txt',
        'silence.txt',
        'nonsilence.txt',
        'wdisambig_phones.int',
        'wdisambig_words.int',
        'mfcc_hires.conf',
        'mfcc.conf',
        'ivector_extractor.conf',
        'splice.conf',
        'online_cmvn.conf',
        'final.mat',
        'global_cmvn.stats',
        'final.dubm',
        'final.ie',
    ]
    nonterminals = list(Compiler.nonterminals)

    for filename in filenames:
        path = find_file(src_dir, filename)
        if path is None:
            _log.error("cannot find %r in %r", filename, model_dir)
            continue
        _log.info("copying %r to %r", path, model_dir)
        shutil.copy(path, model_dir)

    _log.info("converting %r in %r", 'phones.txt', model_dir)
    lines, highest_symbol = augment_phones_txt.read_phones_txt(os.path.join(model_dir, 'phones.txt'))
    augment_phones_txt.write_phones_txt(lines, highest_symbol, nonterminals, os.path.join(model_dir, 'phones.txt'))

    _log.info("converting %r in %r", 'words.txt', model_dir)
    lines, highest_symbol = augment_words_txt.read_words_txt(os.path.join(model_dir, 'words.txt'))
    # FIXME: leave space for adding words later
    augment_words_txt.write_words_txt(lines, highest_symbol, nonterminals, os.path.join(model_dir, 'words.txt'))

    with open(os.path.join(model_dir, 'nonterminals.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(nonterm + '\n' for nonterm in nonterminals)

    # add nonterminals to align_lexicon.int
    
    # fix L_disambig.fst: construct lexiconp_disambig.txt ...


########################################################################################################################

def str_space_join(iterable):
    return u' '.join(text_type(elem) for elem in iterable)

def base_filepath(filepath):
    root, ext = os.path.splitext(filepath)
    return root + '.base' + ext

def verify_files_exist(*filenames):
    return False
