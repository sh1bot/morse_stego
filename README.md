# morse_stego

Hide a short string inside plausible language-model text, then prove it back
out. The cover text is generated so that each whitespace *word's* last letter
spells one Morse symbol of your message — vowel-final = dot, consonant-final =
dash, `y`-final/punctuation = gap — followed by a sentence-ender. Reversing the
words recovers the Morse, which decodes back to your string. Because a symbol
rides on a whole word — which may be several LM tokens — the reversal reads the
plain string and never has to reproduce the model's tokenization.

```
text  --encode-->  morse  --constrained generate-->  LM cover text
      <--decode--  morse  <--reverse-------------
```

The generator is a backtracking constrained decoder: a best-first walk over the
LM's tokens that builds one word at a time under the per-word constraint,
backpedaling to re-pick earlier tokens whenever a position dead-ends, so the
whole message (trailing ender included) is satisfied rather than greedily
stranded.

Everything lives in one file, `morse_stego.py`: the Morse codec, the model
loader, the decoder, and the CLI.

## Run

```
python3 morse_stego.py "SOS"
python3 morse_stego.py "hello world" --prompt "The weather today is"
python3 morse_stego.py "secret" --floor -18 --top-k 300
```

Exit code `0` on a verified round trip, `1` if generation is infeasible or the
input has nothing encodable, `2` on a round-trip mismatch.

Flags: `--prompt` (seed text), `--model` (HF hub id or a local model directory;
or set `MORSE_MODEL`), `--floor` (min per-token logprob; lower = more
permissive), `--top-k` (candidates per position), `--budget` (max backtracking
steps).

To run against real weights with no network — e.g. after copying a `distilgpt2`
snapshot onto the machine — point `--model` at the directory:

```
python3 morse_stego.py "secret message" --model /path/to/distilgpt2
```

## Requirements

`torch`, `transformers`, `tokenizers` — but only the generation step imports
them, so `--help` and the Morse codec stay import-light. With Hugging Face
access the tool loads distilgpt2 and the cover text reads like (rough) English.
The local HF cache is tried first with no network call, so once distilgpt2 has
been fetched later runs load it silently — set `HF_HOME` to a persistent
directory to keep that cache across throwaway environments.

Offline, it falls back to a small GPT-2 trained on the fly (see
`offline_model.py`), on a "word-ending pangram" corpus — short words grouped so
every final-letter class (vowel, consonant, `y`) is covered. The text is
gibberish but the constraint and the round trip are exactly the same. Training
takes a few seconds and is cached under `~/.cache/morse_stego`, so only the first
offline run pays for it. The offline model is weak, so longer strings need a
narrower, deeper search: e.g. `"SOS" --top-k 50 --budget 400000`. In general,
raise `--top-k`/lower `--floor` for the real model and lower `--top-k`/raise
`--budget` for the offline one.
