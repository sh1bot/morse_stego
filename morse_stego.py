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

The generator is a backtracking constrained decoder over a real LM: a best-first
walk over the model's tokens that builds one word at a time, backpedaling to
re-pick earlier tokens whenever a position dead-ends. A word is held whole as it
is built (so wordtomorse can weigh more than the last letter in future) and is
capped at a few tokens -- without that cap the search tunnels into absurdly long
words chasing a required final letter instead of trying a different word.

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

# A HF hub id or a local directory of an uploaded model. Point MORSE_MODEL (or
# --model) at a local model snapshot to run against real weights with no network.
MODEL_NAME = os.environ.get("MORSE_MODEL", "distilgpt2")
_MODEL = None


def get_model():
    """Return (tokenizer, model), loading once. A pretrained model when Hugging
    Face is reachable (or a local --model directory), else the trained tiny GPT-2
    from offline_model -- only the text quality differs, the constraints are
    enforced identically either way.

    The local Hugging Face cache is tried first with no network call, so once a
    model has been fetched later runs load it silently (no repeated download or
    rate-limit chatter). Set HF_HOME to a persistent directory to keep that cache
    across ephemeral environments; or point --model at a local snapshot."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:                                         # 1) local cache/dir -- no network
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
        print(f"[loaded {MODEL_NAME} from local files]\n")
    except Exception:
        try:                                     # 2) download, populating the cache
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            print(f"[downloaded {MODEL_NAME} to cache]\n")
        except Exception as e:                   # 3) offline trained tiny GPT-2
            from offline_model import load_offline_model
            print(f"[no HF access ({type(e).__name__}); using offline tiny GPT-2 — "
                  f"text is gibberish but constraints are real]\n")
            tok, model = load_offline_model()
    model.eval()
    _MODEL = (tok, model)
    return _MODEL


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
        logits = model(torch.tensor([ids])).logits[0, -1]
    return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Backtracking constrained decoder.
# --------------------------------------------------------------------------- #

def backtrack(prompt, accept, done, start_state, floor=-12.0, top_k=50,
              budget=20_000, max_tokens=256, seed=None):
    """Best-first DFS with backtracking on the real LM under a plausibility floor.

    accept(state, token_text) -> next_state or None
        Successor state if `token_text` may legally follow, else None (reject).
        The caller threads whatever it needs through `state` -- here: which word
        we are on and the whole word in progress.
    done(state) -> bool
        True when `state` is a complete, valid message; the search stops there.
    floor : reject any candidate whose next-token logprob is below this; a
            position with no surviving candidate is a dead-end to back out of.
    top_k : how many of the vocab's best tokens to consider per position (the
            real vocab is ~50k; we never need the long tail).
    seed  : if given, perturb the candidate ordering with seeded Gumbel noise --
            i.e. sample the walk from the model instead of taking the strict
            best-first order, giving a different (reproducible) cover per seed.

    Returns (text, cover, total_logp, ok), where `text` is the full decoding
    (prompt included) and `cover` is just the generated continuation -- the part
    that carries the message. Finds the first legal sequence in the chosen order.
    """
    tok, _ = get_model()
    banned = set(tok.all_special_ids or [])  # never weave EOS/pad/etc into a word
    rng = random.Random(seed) if seed is not None else None
    start = tok(prompt, return_tensors=None)["input_ids"]
    ids = list(start)
    cum = [0.0]
    states = [start_state]

    def _ret(ok):
        return tok.decode(ids), tok.decode(ids[len(start):]), cum[-1], ok

    def candidates_at(state):
        if len(ids) - len(start) >= max_tokens:
            return []                        # too long -> treat as a dead-end
        lp = _next_logprobs(ids)
        topv, topi = lp.topk(min(top_k, lp.shape[-1]))
        cs = []
        for v, i in zip(topv.tolist(), topi.tolist()):
            if v < floor or i in banned:
                continue                     # topk is best-first; rest only lower
            ns = accept(state, _text(i))
            if ns is not None:
                cs.append((i, v, ns))
        if rng is not None:                  # Gumbel-top-k: order = a sample
            u = lambda: min(max(rng.random(), 1e-9), 1 - 1e-9)
            cs.sort(key=lambda c: c[1] - math.log(-math.log(u())), reverse=True)
        return cs

    stack = [candidates_at(states[-1])]      # untried candidates for each position
    steps = 0
    while True:
        if (steps := steps + 1) > budget:
            return _ret(False)
        frame = stack[-1]
        if not frame:                        # dead-end -> backpedal one level
            if len(ids) == len(start):
                return _ret(False)           # backed past prompt: infeasible
            ids.pop(); cum.pop(); states.pop(); stack.pop()
            continue
        t, lp, ns = frame.pop(0)             # take & consume best remaining sibling
        ids.append(t); cum.append(cum[-1] + lp); states.append(ns)
        if done(ns):
            return _ret(True)
        stack.append(candidates_at(ns))


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def hide(secret, prompt="The weather today is", floor=-18.0, top_k=200,
         budget=50_000, cap=2, seed=None):
    """Encode `secret` to morse and generate cover text whose words spell it.

    The constraint runs per *word*: word i must have wordtomorse(word) == morse[i].
    The search state carries the index of the word being built, the whole word so
    far, and how many tokens it has taken; a word commits when the next word opens
    (or the ender closes the last). `cap` bounds tokens per word -- two is enough
    for "modest"+"ly" shapes and keeps the search from tunnelling into long words.

    Returns (normalized_secret, morse, text, cover, logprob, ok), where `text` is
    the full sentence and `cover` is the generated continuation to reverse."""
    norm = normalize(secret)
    morse = text_to_morse(norm)              # 3-space word sep -> only . - and spaces
    m = len(morse)
    if m == 0:
        return norm, morse, prompt, "", 0.0, False

    # state = (word_index, word_in_progress, tokens_in_word, finished)
    def accept(state, token_text):
        wi, cur, ntok, fin = state
        if fin:
            return None
        cls = _classify(token_text)
        if cls is None:
            return None
        kind, core = cls
        if kind == "ender":                  # closes the last word, ends sentence
            if cur and wi == m - 1 and wordtomorse(cur) == morse[wi]:
                return (wi, cur, ntok, True)
            return None
        if kind == "cont":                   # extend the current word (up to cap)
            if not cur or ntok >= cap:
                return None
            return (wi, cur + core, ntok + 1, False)
        # kind == "start": open a word (committing the previous one first)
        if not cur:                          # the very first word of the cover
            return (wi, core, 1, False)
        if wordtomorse(cur) != morse[wi] or wi + 1 > m - 1:
            return None                       # wrong symbol, or no word slot left
        return (wi + 1, core, 1, False)

    text, cover, logp, ok = backtrack(
        prompt, accept, lambda s: s[3], start_state=(0, "", 0, False),
        floor=floor, top_k=top_k, budget=budget, max_tokens=m * cap + 8, seed=seed)
    return norm, morse, text, cover, logp, ok


def highlight_symbol_char(word):
    """Render a word with the character that decides its morse symbol -- the last
    non-whitespace char, which is what wordtomorse keys on -- shown in bold yellow.

    Handy for seeing why a word counts as dot/dash/gap: punctuation, 'y' and 'g'
    all land in the gap class, so a word ending in ',' or 'y' fills a space."""
    hl, rst = "\033[1;33m", "\033[0m"
    core = word.rstrip()
    if not core.strip():                         # whitespace/empty -> gap, no char
        return f"[{word}]"
    i = len(core) - 1                            # index of last non-whitespace char
    return f"[{word[:i]}{hl}{word[i]}{rst}{word[i + 1:]}]"


def main(argv=None):
    global MODEL_NAME
    p = argparse.ArgumentParser(description="Hide a string in LM text via morse, then verify it decodes back.")
    p.add_argument("text", help="the string to hide (letters/digits; case & punctuation are normalized away)")
    p.add_argument("--prompt", default="The weather today is", help="seed prompt for the cover text")
    p.add_argument("--model", default=MODEL_NAME, help="HF hub id or local model directory (or set MORSE_MODEL)")
    p.add_argument("--floor", type=float, default=-18.0, help="min per-token logprob (lower = more permissive)")
    p.add_argument("--top-k", type=int, default=200, help="candidate tokens considered per position")
    p.add_argument("--budget", type=int, default=50_000, help="max backtracking steps before giving up")
    p.add_argument("--cap", type=int, default=2, help="max LM tokens per word (bounds the search)")
    p.add_argument("--seed", type=int, default=None, help="seed to vary the cover text (reproducible)")
    args = p.parse_args(argv)
    MODEL_NAME = args.model

    norm, morse, text, cover, logp, ok = hide(
        args.text, prompt=args.prompt, floor=args.floor, top_k=args.top_k,
        budget=args.budget, cap=args.cap, seed=args.seed)

    print(f"input       : {args.text!r}")
    print(f"normalized  : {norm!r}")
    print(f"morse       : {morse!r}")
    if not norm:
        print("\nNothing encodable in that input (need letters or digits).")
        return 1
    if not ok:
        print("\nINFEASIBLE: could not spell that message with this model/prompt.")
        print("Try a shorter string, a lower --floor, or a higher --top-k.")
        return 1

    print(f"\ncover text  : {text!r}")
    words = message_words(cover)
    print(f"logprob     : {logp:.2f}   ({len(words)} words)")

    recovered_morse = cover_to_morse(cover)     # OUTPUT words -> morse (drops ender)
    decoded = morse_to_text(recovered_morse)    # morse -> text
    print("\nwords          : " + " ".join(highlight_symbol_char(w) for w in words))
    print(f"reversed morse : {recovered_morse!r}")
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
