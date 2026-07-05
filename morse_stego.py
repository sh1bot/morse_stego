"""
morse_stego -- hide a string inside plausible LM text, then prove it back out.

Pipeline (all four stages on one command):

    text  --encode-->  morse  --constrained generate-->  LM cover text
          <--decode--  morse  <--reverse-------------

The cover text is generated so that each token's "last letter" spells one morse
symbol of your string (vowel-final = dot, consonant-final = dash, y-final/space
= gap), with a trailing sentence-ender. We then reverse the tokens back to morse
and decode, and assert the result equals your (normalized) input.

The generator is a backtracking constrained decoder over a real LM (distilgpt2,
or a tiny random GPT-2 offline): a best-first walk over the model's tokens under
the per-symbol constraint that backpedals and re-picks earlier tokens whenever a
position dead-ends, so the whole message (ender included) is satisfied rather
than greedily stranded.

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
import os
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
    """Map a token to its morse symbol: '.', '-', or ' '.

    Dot   = ends in a vowel.
    Dash  = ends in a consonant EXCEPT 'y' and 'g'.
    Space = ends in 'y' or 'g', or is empty / whitespace / punctuation.

    'y' and 'g' both fall through to the gap so that spaces -- which are under
    pressure from the three-space word separator -- have more ways to be spelled.
    """
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


def tokens_to_morse(pieces, strip_ender=True):
    """Reverse generator output back to the morse string.

    pieces      : the list of decoded generated tokens (no prompt), as returned
                  by backtrack(). Working on the per-token pieces (not the
                  decoded text) is what makes the reversal exact -- decoding then
                  re-tokenizing would not preserve token boundaries.
    strip_ender : drop a trailing sentence-ender token (the postcondition slot);
                  it is not part of the morse message.
    """
    toks = list(pieces)
    if strip_ender and toks and toks[-1].strip() in ENDERS:
        toks = toks[:-1]
    return "".join(wordtomorse(t) for t in toks)


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

def backtrack(prompt, length, allowed, floor=-12.0, top_k=50, budget=20_000):
    """Best-first DFS with backtracking on the real LM under a plausibility floor.

    allowed(token_text, step) -> bool
    floor : reject any candidate whose next-token logprob is below this; a
            position with no surviving candidate is a dead-end to back out of.
    top_k : how many of the vocab's best tokens to consider per position (the
            real vocab is ~50k; we never need the long tail).

    Returns (text, pieces, total_logp, ok), where `pieces` is the list of
    decoded generated tokens. Finds the first full-length legal sequence in
    best-first order -- keeps the high-plausibility choices it can and only
    rewrites the ones that led into a corner.
    """
    tok, _ = get_model()
    start = tok(prompt, return_tensors=None)["input_ids"]
    ids = list(start)
    cum = [0.0]
    _ret = lambda ok: (tok.decode(ids), [_text(i) for i in ids[len(start):]], cum[-1], ok)

    def candidates_at():
        step = len(ids) - len(start)         # tokens generated so far
        lp = _next_logprobs(ids)
        topv, topi = lp.topk(min(top_k, lp.shape[-1]))
        cs = [(i, v) for v, i in zip(topv.tolist(), topi.tolist())
              if v >= floor and allowed(_text(i), step)]
        return cs                            # topk is already best-first

    stack = [candidates_at()]                # untried candidates for each position
    steps = 0
    while len(ids) - len(start) < length:
        if (steps := steps + 1) > budget:
            return _ret(False)
        frame = stack[-1]
        if not frame:                        # dead-end -> backpedal one level
            if len(ids) == len(start):
                return _ret(False)           # backed past prompt: infeasible
            ids.pop(); cum.pop(); stack.pop()
            continue
        t, lp = frame.pop(0)                 # take & consume best remaining sibling
        ids.append(t); cum.append(cum[-1] + lp)
        stack.append(candidates_at())
    return _ret(True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def hide(secret, prompt="The weather today is", floor=-18.0, top_k=200,
         budget=50_000):
    """Encode `secret` to morse and generate cover text whose tokens spell it.

    Returns (normalized_secret, morse, cover_text, pieces, logprob, ok)."""
    norm = normalize(secret)
    morse = text_to_morse(norm)              # 3-space word sep -> only . - and spaces

    def constraint(t, step):
        if step >= len(morse):               # postcondition: end the sentence
            return t.strip() in ENDERS
        return wordtomorse(t) == morse[step]

    text, pieces, logp, ok = backtrack(
        prompt, len(morse) + 1, constraint, floor=floor, top_k=top_k, budget=budget)
    return norm, morse, text, pieces, logp, ok


def highlight_symbol_char(piece):
    """Render a generated token with the character that decides its morse symbol
    -- the last non-whitespace char, which is what wordtomorse keys on -- shown
    in bold yellow. A whitespace/empty token has no such char and spells a gap.

    Handy for seeing why a token counts as dot/dash/gap: punctuation, 'y' and 'g'
    all land in the gap class, so a token ending in ',' or '.' fills a space."""
    hl, rst = "\033[1;33m", "\033[0m"
    core = piece.rstrip()
    if not core.strip():                         # whitespace/empty -> gap, no char
        return f"[{piece}]"
    i = len(core) - 1                            # index of last non-whitespace char
    return f"[{piece[:i]}{hl}{piece[i]}{rst}{piece[i + 1:]}]"


def main(argv=None):
    global MODEL_NAME
    p = argparse.ArgumentParser(description="Hide a string in LM text via morse, then verify it decodes back.")
    p.add_argument("text", help="the string to hide (letters/digits; case & punctuation are normalized away)")
    p.add_argument("--prompt", default="The weather today is", help="seed prompt for the cover text")
    p.add_argument("--model", default=MODEL_NAME, help="HF hub id or local model directory (or set MORSE_MODEL)")
    p.add_argument("--floor", type=float, default=-18.0, help="min per-token logprob (lower = more permissive)")
    p.add_argument("--top-k", type=int, default=200, help="candidate tokens considered per position")
    p.add_argument("--budget", type=int, default=50_000, help="max backtracking steps before giving up")
    args = p.parse_args(argv)
    MODEL_NAME = args.model

    norm, morse, cover, pieces, logp, ok = hide(
        args.text, prompt=args.prompt, floor=args.floor, top_k=args.top_k, budget=args.budget)

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

    print(f"\ncover text  : {cover!r}")
    print(f"logprob     : {logp:.2f}   ({len(pieces)} tokens)")

    recovered_morse = tokens_to_morse(pieces)   # OUTPUT -> morse (drops the ender)
    decoded = morse_to_text(recovered_morse)    # morse -> text
    print("\ntokens         : " + " ".join(highlight_symbol_char(p) for p in pieces))
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
