# WeNet Active Grammar

> **Alpha Software -- early development**

> **Python WeNet speech recognition with grammars that can be set active/inactive dynamically at decode-time**

> Python package developed to enable context-based command & control of computer applications, as in the [Dragonfly](https://github.com/dictation-toolbox/dragonfly) speech recognition framework, using the [WeNet](https://github.com/wenet-e2e/wenet) automatic speech recognition engine.

[![Donate](https://img.shields.io/badge/donate-GitHub-pink.svg)](https://github.com/sponsors/daanzu)
[![Donate](https://img.shields.io/badge/donate-Patreon-orange.svg)](https://www.patreon.com/daanzu)
[![Donate](https://img.shields.io/badge/donate-PayPal-green.svg)](https://paypal.me/daanzu)

Requirements:
* Python 3.7+ **x64**
* Platform: **Windows/Linux/MacOS**
* Python package requirements: `cffi`, `numpy`
* WeNet Model (must be "runtime" format)
    <!-- * Several are available ready-to-go on this project's [releases page](https://github.com/daanzu/wenet_active_grammar/releases/tag/models) and below. -->

Models:

| Model | Download Size |
|--------|--------|
| [gigaspeech_20210728_u2pp_conformer](https://github.com/daanzu/wenet_stt_python/releases/download/models/gigaspeech_20210728_u2pp_conformer.zip) | 549 MB |
| [gigaspeech_20210811_conformer_bidecoder](https://github.com/daanzu/wenet_stt_python/releases/download/models/gigaspeech_20210811_conformer_bidecoder.zip) | 540 MB |

## Usage

## Installation/Building

<!-- Recommended installation via binary wheel from pip (requires a recent version of pip):

```bash
python -m pip install wenet_active_grammar
``` -->

For details on building from source, see the [Github Actions build workflow](.github/workflows/build.yml).

## Author

* David Zurow ([@daanzu](https://github.com/daanzu))

## License

This project is licensed under the GNU Affero General Public License v3 (AGPL-3.0-or-later). See the [LICENSE file](LICENSE) for details. If this license is problematic for you, please contact me.

## Acknowledgments

* Contains and uses code from [WeNet](https://github.com/wenet-e2e/wenet), licensed under the Apache-2.0 License, and other transitive dependencies (see source).
