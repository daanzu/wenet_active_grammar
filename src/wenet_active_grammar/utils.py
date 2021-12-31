#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

import time
import fnmatch, glob, os
import hashlib, json
import threading

from six import binary_type, text_type

from . import _log, _name, __version__, MODEL_DOWNLOADS


########################################################################################################################

_donation_message_enabled = True
_donation_message = ("Wenet_active_grammar v%s: \n"
    "    If this free, open source engine is valuable to you, please consider donating \n"
    "    https://github.com/daanzu/wenet_active_grammar \n"
    "    Disable message by calling `wenet_active_grammar.disable_donation_message()`") % __version__

def show_donation_message():
    if _donation_message_enabled:
        print(_donation_message)
        disable_donation_message()

def disable_donation_message():
    global _donation_message_enabled
    _donation_message_enabled = False


########################################################################################################################

def clock():
    return time.clock()


########################################################################################################################

def touch_file(filename):
    with open(filename, 'ab'):
        os.utime(filename, None)  # Update timestamps

def clear_file(filename):
    with open(filename, 'wb'):
        pass

symbol_table_lookup_cache = dict()

def symbol_table_lookup(filename, input):
    """
    Returns the RHS corresponding to LHS == ``input`` in symbol table in ``filename``.
    """
    cached = symbol_table_lookup_cache.get((filename, input))
    if cached is not None:
        return cached
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2 and input == tokens[0]:
                try:
                    symbol_table_lookup_cache[(filename, input)] = int(tokens[1])
                    return int(tokens[1])
                except Exception as e:
                    symbol_table_lookup_cache[(filename, input)] = tokens[1]
                    return tokens[1]
        return None

def load_symbol_table(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [[int(token) if token.isdigit() else token for token in line.strip().split()] for line in f]

def find_file(directory, filename, required=False, default=False):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, filename))
    if matches:
        matches.sort(key=len)
        _log.log(8, "%s: find_file found file %r", _name, matches[0])
        return matches[0]
    else:
        _log.log(8, "%s: find_file cannot find file %r in %r (or subdirectories)", _name, filename, directory)
        if required:
            raise IOError("cannot find file %r in %r" % (filename, directory))
        if default == True:
            return os.path.join(directory, filename)
        return None

def is_file_up_to_date(filename, *parent_filenames):
    if not os.path.exists(filename): return False
    for parent_filename in parent_filenames:
        if not os.path.exists(parent_filename): return False
        if os.path.getmtime(filename) < os.path.getmtime(parent_filename): return False
    return True


########################################################################################################################

class FSTFileCache(object):

    def __init__(self, cache_filename, tmp_dir=None, dependencies_dict=None, invalidate=False):
        """
        Stores mapping filename -> hash of its contents/data, to detect when recalculaion is necessary. Assumes file is in model_dir.
        Also stores an entry ``dependencies_list`` listing filenames of all dependencies.
        FST files are a special case: they aren't stored in the cache object, because their filename is itself a hash of its content mixed with a hash of its dependencies.
        If ``invalidate``, then initialize a fresh cache.
        """

        self.cache_filename = cache_filename
        self.tmp_dir = tmp_dir
        if dependencies_dict is None: dependencies_dict = dict()
        self.dependencies_dict = dependencies_dict
        self.lock = threading.Lock()

        try:
            self._load()
        except Exception as e:
            _log.info("%s: failed to load cache from %r", self, cache_filename)
            self.cache = None

        must_reset_cache = False
        if invalidate:
            _log.debug("%s: forced invalidate", self)
            must_reset_cache = True
        elif self.cache is None:
            _log.debug("%s: could not load cache", self)
            must_reset_cache = True
        elif self.cache.get('version') != __version__:
            _log.debug("%s: version changed", self)
            must_reset_cache = True
        elif sorted(self.cache.get('dependencies_list', list())) != sorted(dependencies_dict.keys()):
            _log.debug("%s: list of dependencies has changed", self)
            must_reset_cache = True
        elif any(not self.file_is_current(path)
                for (name, path) in dependencies_dict.items()
                if path and os.path.isfile(path)):
            _log.debug("%s: any of the dependencies files' contents (as stored in cache) has changed", self)
            must_reset_cache = True

        if must_reset_cache:
            # Then reset cache
            _log.info("%s: version or dependencies did not match cache from %r; initializing empty", self, cache_filename)
            self.cache = dict({ 'version': text_type(__version__) })
            self.cache_is_new = True
            self.update_dependencies()
            self.save()

    def _load(self):
        with open(self.cache_filename, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)
        self.cache_is_new = False
        self.dirty = False

    dependencies_hash = property(lambda self: self.cache['dependencies_hash'])

    def save(self):
        with open(self.cache_filename, 'w', encoding='utf-8') as f:
            # https://stackoverflow.com/a/14870531
            f.write(json.dumps(self.cache, ensure_ascii=False))
        self.dirty = False

    def update_dependencies(self):
        dependencies_dict = self.dependencies_dict
        for (name, path) in dependencies_dict.items():
            if path and os.path.isfile(path):
                self.add_file(path)
        self.cache['dependencies_list'] = sorted(dependencies_dict.keys())  # list
        self.cache['dependencies_hash'] = self.hash_data([self.cache.get(path) for (key, path) in sorted(dependencies_dict.items())])

    def invalidate(self, filename=None):
        if filename is None:
            _log.info("%s: invalidating all file entries in cache", self)
            # Does not invalidate dependencies!
            self.cache = { key: self.cache[key]
                for key in ['version', 'dependencies_list', 'dependencies_hash'] + self.cache['dependencies_list']
                if key in self.cache }
            self.dirty = True
            if self.tmp_dir is not None:
                for filename in glob.glob(os.path.join(self.tmp_dir, '*.fst')):
                    os.remove(filename)
        elif filename in self.cache:
            _log.info("%s: invalidating cache entry for %r", self, filename)
            del self.cache[filename]
            self.dirty = True

    def hash_data(self, data, mix_dependencies=False):
        if not isinstance(data, binary_type):
            if not isinstance(data, text_type):
                data = text_type(data)
            data = data.encode('utf-8')
        hasher = hashlib.md5()
        if mix_dependencies:
            hasher.update(self.dependencies_hash.encode('utf-8'))
        hasher.update(data)
        return text_type(hasher.hexdigest())

    def add_file(self, filepath, data=None):
        # Assumes file is a root dependency
        if data is None:
            with open(filepath, 'rb') as f:
                data = f.read()
        filename = os.path.basename(filepath)
        self.cache[filename] = self.hash_data(data)
        self.dirty = True

    def contains(self, filename, data):
        return (filename in self.cache) and (self.cache[filename] == self.hash_data(data))

    def file_is_current(self, filepath, data=None):
        """Returns bool whether generic filepath file exists and the cache contains the given data (or the file's current data if none given)."""
        filename = os.path.basename(filepath)
        if self.cache_is_new and filename in self.cache.get('dependencies_list', list()):
            return False
        if not os.path.isfile(filepath):
            return False
        if data is None:
            with open(filepath, 'rb') as f:
                data = f.read()
        return self.contains(filename, data)

    def fst_is_current(self, filepath, touch=True):
        """Returns bool whether FST file in directory path exists."""
        result = os.path.isfile(filepath)
        if result and touch:
            touch_file(filepath)
        return result


########################################################################################################################

DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024

def download_model(name, url=None, parent_dir='.', verbose=False):
    import os, zipfile
    from urllib.request import urlopen

    if url is None:
        url = MODEL_DOWNLOADS[name]

    filename = os.path.join(parent_dir, name + '.zip')
    if os.path.exists(filename):
        raise FileExistsError(filename)
    if os.path.exists(name):
        raise FileExistsError(name)

    if verbose:
        print("Downloading model '%s'..." % name)
    with urlopen(url) as response:
        with open(filename, 'wb') as f:
            response_length = int(response.getheader('Content-Length'))  # Don't trust response.length!
            bytes_read = 0
            report_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for chunk in iter(lambda: response.read(DOWNLOAD_CHUNK_SIZE), b''):
                f.write(chunk)
                bytes_read += len(chunk)
                if verbose:
                    print('.', end='', flush=True)
                    while report_percentages and bytes_read >= report_percentages[0] * response_length / 100:
                        print(' %d%% ' % report_percentages.pop(0), end='', flush=True)

    if verbose:
        print("Done!")
        print("Extracting...")
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(parent_dir)
    if verbose:
        print("Done!")
        print("Removing zip file...")
    os.remove(filename)
    if verbose:
        print("Done!")
