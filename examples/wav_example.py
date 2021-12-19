import logging, os
import wenet_active_grammar

logging.basicConfig(level=20)
model_dir = '../tests/model'

##### Set up grammar compiler & decoder

compiler = wenet_active_grammar.Compiler(model_dir=model_dir)
decoder = compiler.init_decoder()

##### Set up a rule

rule = wenet_active_grammar.WenetRule(compiler, 'TestRule')
fst = rule.fst

# Construct grammar in a FST
previous_state = fst.add_state(initial=True)
for word in "i will order the".split():
    state = fst.add_state()
    fst.add_arc(previous_state, state, word)
    if word == 'the':
        # 'the' is optional, so we also allow an epsilon (silent) arc
        fst.add_arc(previous_state, state, None)
    previous_state = state
final_state = fst.add_state(final=True)
for word in ['egg', 'bacon', 'sausage']: fst.add_arc(previous_state, final_state, word)
fst.add_arc(previous_state, final_state, 'spam', weight=8)  # 'spam' is much more likely
fst.add_arc(final_state, previous_state, None)  # Loop back, with an epsilon (silent) arc

rule.compile()
rule.load()

##### You could add many more rules...

##### Perform decoding on wav file

import wave

with wave.open('test_it-depends-on-the-context.wav', 'rb') as f:
    wav_data = f.readframes(f.getnframes())

decoder.set_grammars_activity([True])
decoder.decode(wav_data, finalize=True)
result, final = decoder.get_result(final=True)
assert final
print(repr(result))
