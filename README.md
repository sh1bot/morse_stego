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

The generator is a word-level backtracking decoder. For each symbol it builds one
ranked list of candidate words — whole single-token words *and* multi-token
fusions together, scored by cumulative logprob — then walks them best-first,
backing out a whole word when a position dead-ends. Ranking both kinds in one
list is the point: a natural fusion (a word plus a comma, to spell a gap) can
outrank a single word, while a clumsy one (`for`+`no` → `forno`) loses to any
decent whole word. `--cap N` is the longest word (in tokens) the list may reach;
`cap=2` (the default) reads better than `cap=1` because commas and other short
tails make nicer sentences, and the ranking keeps the junk out on its own.

The pure morse codec (no torch) lives in `morse_codec.py`, which is also a
standalone decoder; `morse_stego.py` imports it for generation and validation,
and the offline fallback model lives in `offline_model.py`.

## Decoding

`morse_codec.py` decodes cover text on its own — no model needed:

```
python3 morse_codec.py "a place where they ... more."
python3 morse_codec.py "<full cover>" --prompt "The weather today is"
```

The message rides in the words *after* the generation prompt, so pass `--prompt`
to strip a known seed prefix first (the demo prints the full sentence including
that prompt). `morse_stego.py` validates its own output through the same
functions, so the encode and decode paths can't drift.

## Run

```
python3 morse_stego.py "SOS"
python3 morse_stego.py "hello world" --prompt "The weather today is"
python3 morse_stego.py "secret" --floor -18 --top-k 300
```

Exit code `0` on a verified round trip, `1` if generation is infeasible or the
input has nothing encodable, `2` on a round-trip mismatch.

Flags: `--prompt` (seed text), `--model` (HF hub id or a local model directory;
or set `MORSE_MODEL`), `--device` (`cuda`/`mps`/`cpu`, default auto), `--dtype`
(default auto: bf16/fp16 on GPU, fp32 on CPU), `--trust-remote-code` (for models
that ship custom code), `--floor` (min per-token logprob; lower = more
permissive), `--top-k` (candidates per position), `--budget` (max backtracking
steps), `--cap` (longest word in LM tokens the candidate list may reach; default
2), `--seed` (vary the cover text
reproducibly — same seed, same output; different seed, different cover),
`--temperature` (how bold that variety is; lower = milder and more coherent),
`--count` (print N covers from consecutive seeds, all from one model load, to
pick one you like).

## Choosing a model

`--model` takes any Hugging Face causal-LM id or a local directory — switching
models is just the normal HF interface, with the weights cached under `HF_HOME`.
A better *base* model makes the cover text far more lucid; pick a **byte-level
BPE** family (GPT-2/Neo, **Qwen2.5**, **Llama-3**) so word starts carry a leading
space — the tool warns if a tokenizer (some SentencePiece ones) doesn't. Prefer a
base model over an instruct/chat one, since the cover text is a free continuation.

```
python3 morse_stego.py "secret message" --model Qwen/Qwen2.5-1.5B      # lucid, CPU-friendly
python3 morse_stego.py "secret message" --model Qwen/Qwen2.5-7B --device cuda
python3 morse_stego.py "secret message" --model /path/to/local/model  # offline, real weights
```

Bigger models are more fluent but the search makes many forward passes
(~`4·cap` per word), so use a GPU past ~3B, and note that a lucid model often
reads well at `--cap 1` (one pass per word, much faster).

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
