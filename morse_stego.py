"""
morse_stego -- hide a string inside plausible LM text, then prove it back out.

Pipeline (all four stages on one command):

    text  --encode-->  morse  --constrained generate-->  LM cover text
          <--decode--  morse  <--reverse-------------

The cover text is generated so that each whitespace *word's* "last letter" spells
one morse symbol of your string (vowel-final = dot, consonant-final = dash,
y-final/punctuation = gap), with a trailing sentence-ender. We then reverse the
words back to morse and decode, and assert the result equals your (normalized)
input. Because a symbol rides on a whole word -- which may be several LM tokens --
the reversal reads the plain string and never needs the model's tokenization.

The generator is a backtracking constrained decoder over a real LM (distilgpt2,
or a tiny random GPT-2 offline): a best-first walk over the model's tokens that
builds one word at a time under the per-word constraint, backpedaling and
re-picking earlier tokens whenever a position dead-ends, so the whole message
(ender included) is satisfied rather than greedily stranded.

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
    """Map a word to its morse symbol: '.', '-', or ' ', from its last letter.

    Dot   = ends in a vowel.
    Dash  = ends in a consonant EXCEPT 'y'.
    Space = ends in 'y', or is empty / whitespace / punctuation.
    """
    word = word.strip()
    if not word:
        return " "
    last = word[-1].lower()
    if last in "aeiou":
        return "."
    if last in "bcdfghjklmnpqrstvwxz":        # note: no 'y'
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


def cover_to_morse(cover):
    """Reverse the cover text back to its morse string, reading whole words.

    Splits the visible text on whitespace and maps each word's last letter to one
    morse symbol. A word may be any number of LM tokens -- reading words instead
    of tokens is what lets encode and decode meet over the plain string, with no
    token boundaries to preserve. The trailing sentence-ender (the postcondition
    slot, not part of the message) is stripped first.
    """
    cover = cover.rstrip()
    while cover and cover[-1] in ENDERS:
        cover = cover[:-1].rstrip()
    return "".join(wordtomorse(w) for w in cover.split())


def _classify(text):
    """Classify a decoded token as ('start'|'cont'|'ender', core) or None.

    start : begins a new word -- one leading space, then any non-space run.
    cont  : continues the current word -- any non-space run, no leading space.
    ender : a bare sentence-ender ('.', '?', '!'), whichever side the space is on.

    Only the leading space (word boundary) and the absence of internal whitespace
    matter. A word's *content* is otherwise unconstrained: since a symbol is read
    off a word's last letter, every token before the last is free choice, which is
    what keeps the search wide. Tokens with internal or trailing whitespace are the
    only ones rejected, so the decoded text splits back into the same words.
    """
    if not text:
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

MODEL_NAME = "distilgpt2"
_MODEL = None


def get_model():
    """Return (tokenizer, model), loading once. Pretrained distilgpt2 when
    Hugging Face is reachable, else the trained tiny GPT-2 from offline_model --
    only the text quality differs, the constraints are enforced identically.

    The local Hugging Face cache is tried first with no network call, so once
    distilgpt2 has been fetched, later runs load it silently (no repeated
    download or rate-limit chatter). Set HF_HOME to a persistent directory to
    keep that cache across ephemeral environments."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:                                         # 1) local cache only -- no network
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
        print(f"[loaded {MODEL_NAME} from local cache]\n")
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
              budget=20_000, max_tokens=256):
    """Best-first DFS with backtracking on the real LM under a plausibility floor.

    accept(state, token_text) -> next_state or None
        Successor state if `token_text` may legally follow, else None (reject).
        The caller threads whatever it needs through `state` (here: which word
        we are on and the letters of the word in progress).
    done(state) -> bool
        True when `state` is a complete, valid message; the search stops there.
    floor : reject any candidate whose next-token logprob is below this; a
            position with no surviving candidate is a dead-end to back out of.
    top_k : how many of the vocab's best tokens to consider per position (the
            real vocab is ~50k; we never need the long tail).

    Returns (text, cover, total_logp, ok), where `text` is the full decoding
    (prompt included) and `cover` is just the generated continuation -- the part
    that carries the message. Finds the first legal sequence in best-first order.
    """
    tok, _ = get_model()
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
            if v < floor:
                continue                     # topk is best-first; rest only lower
            ns = accept(state, _text(i))
            if ns is not None:
                cs.append((i, v, ns))
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
         budget=50_000):
    """Encode `secret` to morse and generate cover text whose words spell it.

    The constraint runs per *word*: word i must have wordtomorse(word) == morse[i].
    A word is built from one or more tokens, so the search state carries the index
    of the word being built and its letters so far; a word only commits when the
    next word begins (or the ender closes the last one).

    Returns (normalized_secret, morse, text, cover, logprob, ok), where `text` is
    the full sentence and `cover` is the generated continuation to reverse."""
    norm = normalize(secret)
    morse = text_to_morse(norm)              # 3-space word sep -> only . - and spaces
    m = len(morse)
    if m == 0:
        return norm, morse, prompt, "", 0.0, False

    # state = (word_index, letters_of_word_in_progress, finished)
    def accept(state, token_text):
        wi, cur, fin = state
        if fin:
            return None
        cls = _classify(token_text)
        if cls is None:
            return None
        kind, core = cls
        if kind == "ender":                  # closes the last word, ends sentence
            if cur and wi == m - 1 and wordtomorse(cur) == morse[wi]:
                return (wi, cur, True)
            return None
        if kind == "cont":                   # extend the current word
            if not cur:
                return None
            return (wi, cur + core, False)
        # kind == "start": open a word (committing the previous one first)
        if not cur:                          # the very first word of the cover
            return (wi, core, False)
        if wordtomorse(cur) != morse[wi] or wi + 1 > m - 1:
            return None                       # wrong symbol, or no word slot left
        return (wi + 1, core, False)

    text, cover, logp, ok = backtrack(
        prompt, accept, lambda s: s[2], start_state=(0, "", False),
        floor=floor, top_k=top_k, budget=budget, max_tokens=m * 6 + 8)
    return norm, morse, text, cover, logp, ok


def main(argv=None):
    p = argparse.ArgumentParser(description="Hide a string in LM text via morse, then verify it decodes back.")
    p.add_argument("text", help="the string to hide (letters/digits; case & punctuation are normalized away)")
    p.add_argument("--prompt", default="The weather today is", help="seed prompt for the cover text")
    p.add_argument("--floor", type=float, default=-18.0, help="min per-token logprob (lower = more permissive)")
    p.add_argument("--top-k", type=int, default=200, help="candidate tokens considered per position")
    p.add_argument("--budget", type=int, default=50_000, help="max backtracking steps before giving up")
    args = p.parse_args(argv)

    norm, morse, text, cover, logp, ok = hide(
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

    print(f"\ncover text  : {text!r}")
    print(f"logprob     : {logp:.2f}   ({len(cover.split())} words)")

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
