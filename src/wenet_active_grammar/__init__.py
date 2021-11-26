#
# This file is part of wenet_active_grammar.
# (c) Copyright 2021 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE file.
#

_name = 'wenet_active_grammar'
__version__ = '0.1.0'
__author__ = 'David Zurow'
__license__ = 'AGPL-3.0'

MODEL_DOWNLOADS = {
    'gigaspeech_20210728_u2pp_conformer': 'https://github.com/daanzu/wenet_active_grammar/releases/download/models/gigaspeech_20210728_u2pp_conformer.zip',
    'gigaspeech_20210811_conformer_bidecoder': 'https://github.com/daanzu/wenet_active_grammar/releases/download/models/gigaspeech_20210811_conformer_bidecoder.zip'
}

import logging
_log = logging.getLogger('wenet')

class WenetError(Exception):
    pass

from .compiler import Compiler, KaldiRule
from .wrapper import KaldiAgfNNet3Decoder, KaldiLafNNet3Decoder, KaldiPlainNNet3Decoder
from .wfst import WFST
from .plain_dictation import PlainDictationRecognizer
from .utils import disable_donation_message
