"""
morse_stego -- hide a string inside plausible LM text, then prove it back out.

Pipeline (all four stages on one command):

    text  --encode-->  morse  --constrained generate-->  LM cover text
          <--decode--  morse  <--reverse-------------

The cover text is generated so that each whitespace *word's* last letter spells
one morse symbol of your string (vowel-final = dot, consonant-final = dash,
y/g/punctuation-final = gap), with a trailing sentence-ender. We then reverse the
words back to morse and decode, and assert the result equals your (normalized)
input. Because a symbol rides on a whole word -- which may be a couple of LM
tokens -- the reversal reads the plain string and never has to reproduce the
model's tokenization.

The generator is a word-level backtracking decoder over a real LM. For each morse
symbol it builds one ranked list of candidate words -- whole single-token words
AND multi-token fusions together, scored by cumulative logprob -- then walks them
best-first, backing out a whole word when a position dead-ends. Ranking the two
kinds together is the point: a natural fusion (a word plus a comma, to spell a
gap) can outrank a single word, while a clumsy one ("for"+"no" -> "forno") loses
to any decent whole word. --cap is the longest word (in tokens) the list may
reach; cap=2 is a good default because commas and other short tails make better
sentences, and the ranking keeps the junk out on its own.

Usage:
    python3 morse_stego.py "SOS"
    python3 morse_stego.py "hello world" --prompt "The weather today is"
    python3 morse_stego.py "secret" --floor -18 --top-k 300

Exit code 0 on a verified round trip, 1 if generation is infeasible (or nothing
encodable), 2 if the round trip does not match.

Requires torch, transformers, tokenizers -- but only the generation step needs
them; the morse codec and --help stay import-light.
"""

import argparse
import heapq
import math
import os
import random
import re
import sys
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Morse codec -- one symbol mapping shared by encode and decode, so the text
# that drives generation and the parser that reads it back can't drift apart.
# --------------------------------------------------------------------------- #

ENDERS = {".", "?", "!"}

# International Morse: letters + digits.
MORSE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}
MORSE_INV = {code: letter for letter, code in MORSE.items()}


def wordtomorse(word):
    """Map a whole word to its morse symbol: '.', '-', or ' '.

    Dot   = ends in a vowel.
    Dash  = ends in a consonant EXCEPT 'y' and 'g'.
    Space = ends in 'y' or 'g', or is empty / whitespace / punctuation.

    'y' and 'g' both fall through to the gap so that spaces -- which are under
    pressure from the three-space word separator -- have more ways to be spelled.

    The whole word is passed in (not just its last letter) and this is the single
    place both encode and decode decide a symbol, so a future rule can weigh more
    of the word without the two sides drifting apart."""
    word = word.strip()
    if not word:
        return " "
    last = word[-1].lower()
    if last in "aeiou":
        return "."
    if last in "bcdfhjklmnpqrstvwxz":         # note: no 'y' or 'g' (both gaps)
        return "-"
    return " "


def normalize(text):
    """Uppercase and keep only morse-encodable characters, collapsing whitespace
    to single word breaks. This is exactly what a decoded round trip can
    reproduce (morse is case-insensitive and has only the table's characters)."""
    words = []
    for word in text.upper().split():
        kept = "".join(c for c in word if c in MORSE)
        if kept:
            words.append(kept)
    return " ".join(words)


def text_to_morse(text, word_sep="   "):
    """Encode text to a morse string (letters space-separated, words by word_sep).

    The default word_sep is three spaces so the result is only dots, dashes and
    spaces -- reproducible one-token-per-symbol by a constraint that has no way
    to emit a '/'. morse_to_text decodes ' / ' and 3+ spaces alike."""
    words = []
    for word in text.upper().split():
        words.append(" ".join(MORSE.get(ch, "?") for ch in word))
    return word_sep.join(words)


def morse_to_text(morse):
    """Decode a morse string to text. Words split on ' / ' or 3+ spaces; letters
    within a word split on single spaces. Unknown codes -> '?'."""
    words = []
    for word in re.split(r"\s*/\s*|\s{3,}", morse.strip()):
        if not word:
            continue
        words.append("".join(MORSE_INV.get(code, "?") for code in word.split()))
    return " ".join(words)


def message_words(cover):
    """Split cover text into the words that carry the message: whitespace-
    separated, with the trailing sentence-ender (the postcondition slot, not part
    of the message) stripped first."""
    cover = cover.rstrip()
    while cover and cover[-1] in ENDERS:
        cover = cover[:-1].rstrip()
    return cover.split()


def cover_to_morse(cover):
    """Reverse cover text back to its morse string by reading whole words.

    Each whitespace word maps to one morse symbol via wordtomorse. A word may be
    any number of LM tokens -- reading words instead of tokens is what lets encode
    and decode meet over the plain string, with no token boundaries to preserve."""
    return "".join(wordtomorse(w) for w in message_words(cover))


def _classify(text):
    """Classify a decoded token as ('start'|'cont'|'ender', core) or None.

    start : begins a new word -- one leading space, then any non-space run.
    cont  : continues the current word -- any non-space run, no leading space.
    ender : a bare sentence-ender ('.', '?', '!'), whichever side the space is on.

    A word's content is unconstrained apart from the leading space that marks a
    boundary: since a symbol is read off the last letter, every token before the
    last is free. Only tokens with internal/trailing whitespace are rejected, so
    the decoded text splits back into exactly the words we built."""
    if not text or "�" in text:            # reject partial-UTF-8 byte tokens
        return None
    leading = text[:1] == " "
    core = text[1:] if leading else text
    if not core or any(c.isspace() for c in core):
        return None
    if core in ENDERS:
        return ("ender", core)
    return ("start" if leading else "cont", core)


# --------------------------------------------------------------------------- #
# Language model -- loaded lazily so the codec above and --help stay torch-free.
# --------------------------------------------------------------------------- #

# Any Hugging Face causal-LM id or a local model directory works via --model /
# MORSE_MODEL -- switch models entirely through the normal HF interface. A better
# base model (byte-level BPE like Qwen2.5 or Llama-3, so word starts carry a
# leading space) makes the cover text far more lucid; --device / --dtype control
# where and how it runs.
MODEL_NAME = os.environ.get("MORSE_MODEL", "distilgpt2")
MODEL_DEVICE = os.environ.get("MORSE_DEVICE")            # None -> auto-detect
MODEL_DTYPE = os.environ.get("MORSE_DTYPE")              # None -> auto by device
TRUST_REMOTE_CODE = os.environ.get("MORSE_TRUST_REMOTE_CODE", "") not in ("", "0")
_MODEL = None


def _resolve_device():
    if MODEL_DEVICE:
        return MODEL_DEVICE
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Return (tokenizer, model), loading once. Any HF causal LM named by --model /
    MORSE_MODEL (or a local directory), else the trained tiny GPT-2 from
    offline_model -- only the text quality differs, the constraints are enforced
    identically either way.

    The local Hugging Face cache is tried first with no network call, so once a
    model has been fetched later runs load it silently. Set HF_HOME to a
    persistent cache. The model is placed on --device (auto: cuda/mps/cpu) at a
    dtype that suits it (bf16/fp16 on GPU, fp32 on CPU to avoid half-precision CPU
    ops); override with --dtype."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device()
    common = {"trust_remote_code": True} if TRUST_REMOTE_CODE else {}
    load = dict(common)
    if MODEL_DTYPE:
        load["torch_dtype"] = getattr(torch, MODEL_DTYPE)
    elif device != "cpu":                        # half precision only off the CPU
        load["torch_dtype"] = (torch.bfloat16 if device == "cuda"
                               and torch.cuda.is_bf16_supported() else torch.float16)

    def _load(local_only):
        kw = {"local_files_only": True} if local_only else {}
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, **kw, **common)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kw, **load)
        return tok, model

    real = True
    try:                                         # 1) local cache/dir -- no network
        tok, model = _load(True); via = "local files"
    except Exception:
        try:                                     # 2) download, populating the cache
            tok, model = _load(False); via = "download"
        except Exception as e:                   # 3) offline trained tiny GPT-2
            from offline_model import load_offline_model
            print(f"[no HF access ({type(e).__name__}); using offline tiny GPT-2 — "
                  f"text is gibberish but constraints are real]\n")
            tok, model = load_offline_model(); real = False
    if real:
        model.to(device)
        print(f"[loaded {MODEL_NAME} from {via} on {model.device}]\n")
        _warn_if_word_boundaries_unclear(tok)
    model.eval()
    _MODEL = (tok, model)
    return _MODEL


def _warn_if_word_boundaries_unclear(tok):
    """The word split keys on a leading space marking each word start; byte-level
    BPE tokenizers (GPT-2, Qwen2.5, Llama-3) satisfy this, some SentencePiece ones
    do not. Warn rather than fail, since only text quality is at stake."""
    try:
        pieces = [tok.decode([i]) for i in tok(" one two three")["input_ids"]]
    except Exception:
        return
    if not any(p.startswith(" ") for p in pieces):
        print("[warning: this tokenizer doesn't mark word starts with a leading "
              "space, so the word split may misbehave -- prefer a byte-level BPE "
              "model such as Qwen2.5 or Llama-3]\n")


@lru_cache(maxsize=None)
def _text(token_id):
    """Decoded text for a single token id (cached; decoding is the hot cost)."""
    return get_model()[0].decode([token_id])


def _next_logprobs(ids):
    """Next-token log-probabilities over the whole vocab, given a full id context."""
    import torch
    import torch.nn.functional as F
    _, model = get_model()
    with torch.no_grad():
        inp = torch.tensor([ids], device=model.device)
        logits = model(inp).logits[0, -1]
    return F.log_softmax(logits.float(), dim=-1)


# --------------------------------------------------------------------------- #
# Backtracking constrained decoder.
# --------------------------------------------------------------------------- #

def word_candidates(ids, matches, ender, cap, top_k, floor, banned, rng,
                    temperature, max_passes):
    """Ranked candidate *words* for the next position: each (added_ids, logp, word).

    A candidate is a word-start token plus up to cap-1 continuation tokens whose
    assembled form `matches` the position (its morse symbol, or a sentence-ender
    when `ender`). Words of every length are gathered into ONE list and ranked by
    cumulative logprob, so a natural fusion ("outside"+"," for a gap) can outrank
    a single word, while a poor one ("for"+"no") cannot outrank a good single word.
    We deepen (extend the best not-yet-matching openings) up to `max_passes`
    forward passes, so a common position costs a single pass but fusions still get
    a fair hearing."""
    found = []
    heap = [(0.0, 0, [], "")]                # (-logp, tiebreak, added_ids, word)
    tie = passes = 0
    while heap and passes < max_passes:
        neg, _, added, word = heapq.heappop(heap)
        logp, ntok = -neg, len(added)
        topv, topi = _next_logprobs(ids + added).topk(top_k)
        passes += 1
        for v, i in zip(topv.tolist(), topi.tolist()):
            if v < floor or i in banned:
                continue
            cls = _classify(_text(i))
            if cls is None:
                continue
            kind, core = cls
            if ntok == 0:                    # the token that opens the word
                if kind != ("ender" if ender else "start"):
                    continue
                nword = core
            else:                            # a continuation of the opening
                if kind != "cont":
                    continue
                nword = word + core
            nlogp = logp + v
            if matches(nword):
                found.append((added + [i], nlogp, nword))
            elif not ender and ntok + 1 < cap:
                tie += 1
                heapq.heappush(heap, (-nlogp, tie, added + [i], nword))
    if rng is None:
        found.sort(key=lambda c: c[1], reverse=True)
    else:                                    # Gumbel-top-k over whole words, at T
        found.sort(key=lambda c: c[1] / temperature - math.log(-math.log(
            min(max(rng.random(), 1e-9), 1 - 1e-9))), reverse=True)
    return found[:top_k]


def word_search(prompt, morse, cap=1, top_k=200, floor=-18.0, budget=50_000,
                seed=None, temperature=1.0):
    """Word-level best-first DFS. Fill positions 0..m-1 (one word per morse symbol)
    then a trailing sentence-ender, each from word_candidates() ranked by
    cumulative logprob, backtracking a whole word when a position runs dry.

    Returns (text, cover, total_logp, ok); `cover` is the generated continuation."""
    tok, _ = get_model()
    banned = set(tok.all_special_ids or [])
    rng = random.Random(seed) if seed is not None else None
    start = tok(prompt, return_tensors=None)["input_ids"]
    ids = list(start)
    m = len(morse)
    max_passes = 4 * cap                     # 1 opening pass + fusion deepening

    def frame_for(pos):
        if pos < m:
            sym = morse[pos]
            return word_candidates(ids, lambda w: wordtomorse(w) == sym, False,
                                   cap, top_k, floor, banned, rng, temperature,
                                   max_passes)
        return word_candidates(ids, lambda w: w.strip() in ENDERS, True,
                               cap, top_k, floor, banned, rng, temperature,
                               max_passes)

    added, logps, frames = [], [], [frame_for(0)]
    steps = 0
    while True:
        frame = frames[-1]
        if not frame:                        # dead-end -> back out one word
            if not added:
                return tok.decode(start), "", 0.0, False
            for _ in added.pop():
                ids.pop()
            logps.pop()
            frames.pop()
            continue
        if (steps := steps + 1) > budget:
            return tok.decode(ids), tok.decode(ids[len(start):]), sum(logps), False
        add_ids, lp, word = frame.pop(0)
        ids += add_ids
        added.append(add_ids)
        logps.append(lp)
        if len(added) > m:                   # just placed the ender -> done
            return tok.decode(ids), tok.decode(ids[len(start):]), sum(logps), True
        frames.append(frame_for(len(added)))


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def hide(secret, prompt="The weather today is", floor=-18.0, top_k=200,
         budget=50_000, cap=2, seed=None, temperature=1.0):
    """Encode `secret` to morse and generate cover text whose words spell it.

    One morse symbol per whitespace word (its last letter). word_search fills one
    position per symbol, each from a single ranked list of candidate words that
    mixes whole words and multi-token fusions by cumulative logprob -- so a natural
    fusion (a word plus a comma, say) competes fairly and a poor one loses to any
    good single word. `cap` is the longest word (in tokens) the list may reach.

    Returns (normalized_secret, morse, text, cover, logprob, ok), where `text` is
    the full sentence and `cover` is the generated continuation to reverse."""
    norm = normalize(secret)
    morse = text_to_morse(norm)              # 3-space word sep -> only . - and spaces
    if not morse:
        return norm, morse, prompt, "", 0.0, False
    text, cover, logp, ok = word_search(
        prompt, morse, cap=cap, top_k=top_k, floor=floor, budget=budget,
        seed=seed, temperature=temperature)
    return norm, morse, text, cover, logp, ok


def main(argv=None):
    global MODEL_NAME, MODEL_DEVICE, MODEL_DTYPE, TRUST_REMOTE_CODE
    p = argparse.ArgumentParser(description="Hide a string in LM text via morse, then verify it decodes back.")
    p.add_argument("text", help="the string to hide (letters/digits; case & punctuation are normalized away)")
    p.add_argument("--prompt", default="The weather today is", help="seed prompt for the cover text")
    p.add_argument("--model", default=MODEL_NAME, help="HF hub id or local model directory (or set MORSE_MODEL)")
    p.add_argument("--device", default=MODEL_DEVICE, help="cuda / mps / cpu (default: auto-detect)")
    p.add_argument("--dtype", default=MODEL_DTYPE, help="torch dtype, e.g. bfloat16 / float16 / float32 (default: auto)")
    p.add_argument("--trust-remote-code", action="store_true", default=TRUST_REMOTE_CODE, help="allow models that ship custom code")
    p.add_argument("--floor", type=float, default=-18.0, help="min per-token logprob (lower = more permissive)")
    p.add_argument("--top-k", type=int, default=200, help="candidate tokens considered per position")
    p.add_argument("--budget", type=int, default=50_000, help="max backtracking steps before giving up")
    p.add_argument("--cap", type=int, default=2, help="longest word in LM tokens the candidate list may reach")
    p.add_argument("--seed", type=int, default=None, help="seed to vary the cover text (reproducible)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for --seed (lower = milder variety)")
    p.add_argument("--count", type=int, default=1, help="produce N covers (different seeds) to choose from")
    args = p.parse_args(argv)
    MODEL_NAME = args.model
    MODEL_DEVICE = args.device
    MODEL_DTYPE = args.dtype
    TRUST_REMOTE_CODE = args.trust_remote_code

    norm = normalize(args.text)
    morse = text_to_morse(norm)
    print(f"input       : {args.text!r}")
    print(f"normalized  : {norm!r}")
    print(f"morse       : {morse!r}")
    if not norm:
        print("\nNothing encodable in that input (need letters or digits).")
        return 1

    # One seed per variant; consecutive from --seed (or 0) so a run is reproducible.
    count = max(1, args.count)
    if count == 1:
        seeds = [args.seed]
    else:
        base = args.seed if args.seed is not None else 0
        seeds = [base + i for i in range(count)]

    rc = 0
    for n, sd in enumerate(seeds, 1):
        if count > 1:
            print(f"\n--- variant {n}/{count} (seed {sd}) ---")
        _, _, text, cover, logp, ok = hide(
            args.text, prompt=args.prompt, floor=args.floor, top_k=args.top_k,
            budget=args.budget, cap=args.cap, seed=sd,
            temperature=args.temperature)
        rc = max(rc, report(text, cover, logp, morse, norm, ok))
    return rc


def report(text, cover, logp, morse, norm, ok):
    """Print one cover and its round trip; return 0 (pass), 1 (infeasible),
    or 2 (round-trip mismatch)."""
    if not ok:
        print("\nINFEASIBLE: could not spell that message with this model/prompt.")
        print("Try a lower --floor, a higher --top-k, or a higher --cap.")
        return 1

    print(f"\ncover text  : {text!r}")
    words = message_words(cover)
    print(f"logprob     : {logp:.2f}   ({len(words)} words)")

    recovered_morse = cover_to_morse(cover)     # OUTPUT words -> morse (drops ender)
    decoded = morse_to_text(recovered_morse)    # morse -> text
    print(f"\nreversed morse : {recovered_morse!r}")
    print(f"decoded text   : {decoded!r}")

    morse_ok = recovered_morse == morse
    text_ok = decoded == norm
    print(f"\nmorse round-trips : {morse_ok}")
    print(f"text round-trips  : {text_ok}")
    if morse_ok and text_ok:
        print("\nPASS: cover text decodes back to the input.")
        return 0
    print("\nFAIL: round trip did not match.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
