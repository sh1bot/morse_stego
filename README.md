# morse_stego

Hide a short string inside plausible language-model text, then prove it back
out. The cover text is generated so that each whitespace *word's* last letter
spells one Morse symbol of your message — vowel-final = dot, consonant-final =
dash, `y`/`g`/punctuation-final = gap — followed by a sentence-ender. Reversing
the words recovers the Morse, which decodes back to your string. Because a symbol
rides on a whole word — which may be a couple of LM tokens — the reversal reads
the plain string and never has to reproduce the model's tokenization.

```
text  --encode-->  morse  --constrained generate-->  LM cover text
      <--decode--  morse  <--reverse-------------
```

The generator is a backtracking constrained decoder: a best-first walk over the
LM's tokens that builds one word at a time, backpedaling to re-pick earlier
tokens whenever a position dead-ends. Each word is held whole as it is built (so
the symbol rule can weigh more than the last letter later). By default a word is
a single token — one clean whole word per symbol — which on a real model encodes
a phrase like `"secret message"` in well under a second. `--cap N` lets a word
span up to N tokens to hit a required final letter when a message is otherwise
infeasible, but that glues tokens into non-words (`for`+`no` → `forno`), so use
the lowest cap that still works.

The codec, model loader, decoder, and CLI live in `morse_stego.py`; the offline
fallback model lives in `offline_model.py`.

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
steps), `--cap` (max LM tokens per word; 1 = clean whole words, raise if
infeasible), `--seed` (vary the cover text
reproducibly — same seed, same output; different seed, different cover),
`--count` (print N covers from consecutive seeds, all from one model load, to
pick one you like).

To run against real weights with no network — e.g. after copying a model
snapshot onto the machine — point `--model` at the directory:

```
python3 morse_stego.py "secret message" --model /path/to/distilgpt2
```

`fetch_tinystories.py` collates a small real model (`roneneldan/TinyStories-3M`,
~12 MB) plus its tokenizer into one self-contained folder, so anyone can
reproduce the same test on a machine with Hugging Face access:

```
python3 fetch_tinystories.py            # -> ./tinystories_3m/
python3 morse_stego.py "secret message" --model ./tinystories_3m
```

## Requirements

`torch`, `transformers`, `tokenizers` — but only the generation step imports
them, so `--help` and the Morse codec stay import-light. With Hugging Face
access the tool loads distilgpt2 and the cover text reads like (rough) English.
The local HF cache is tried first with no network call, so once distilgpt2 has
been fetched later runs load it silently — set `HF_HOME` to a persistent
directory to keep that cache across throwaway environments.

Offline (no network and nothing cached), it falls back to a small GPT-2 trained
on the fly (see `offline_model.py`) on a "word-ending pangram" corpus — short
words grouped so every final-letter class (vowel, consonant, `y`) is covered.
The text is gibberish but the constraint and the round trip are exactly the same.
Training takes a few seconds and is cached under `~/.cache/morse_stego`, so only
the first offline run pays for it. The offline model is weak, so long strings may
still report `INFEASIBLE`; a real model (via `--model`) is what handles phrases.
