"""
morse_codec -- the pure morse codec shared by encode and decode, plus a
standalone tool to decode cover text back to the hidden string.

No torch or transformers here: this is the import-light half that both the
generator (morse_stego.py, which imports these functions) and anyone who just
wants to read a cover string can use. The single symbol mapping (wordtomorse)
lives here so the text that drives generation and the parser that reads it back
can never drift apart.

    cover text  --cover_to_morse-->  morse  --morse_to_text-->  hidden string

The cover carries the message only in the words *after* the generation prompt --
those seed words map to symbols too, so pass the prompt to strip it first.

As a tool:

    python3 morse_codec.py "a place where they ... more."
    python3 morse_codec.py "The weather today is a place ... more." --prompt "The weather today is"
    echo "<cover text>" | python3 morse_codec.py
"""

import argparse
import re
import sys

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


def strip_prompt(cover, prompt):
    """Drop a known seed prompt from the front of the cover, if present, so only
    the message-bearing words remain."""
    prompt = prompt.strip()
    if prompt and cover.strip().startswith(prompt):
        return cover.strip()[len(prompt):]
    return cover


def decode_cover(cover, prompt=""):
    """Cover text -> hidden string, in one call (strip the prompt, then
    cover_to_morse, then morse_to_text)."""
    return morse_to_text(cover_to_morse(strip_prompt(cover, prompt)))


def main(argv=None):
    """Decode cover text from the command line (or stdin) back to the string."""
    p = argparse.ArgumentParser(description="Decode cover text back to the hidden string.")
    p.add_argument("cover", nargs="*", help="cover text (or pipe it on stdin)")
    p.add_argument("--prompt", default="", help="seed prompt to strip from the front first")
    args = p.parse_args(argv)
    cover = " ".join(args.cover) if args.cover else sys.stdin.read()
    if not cover.strip():
        p.print_usage()
        return 1
    cover = strip_prompt(cover, args.prompt)
    morse = cover_to_morse(cover)
    print(f"morse   : {morse!r}")
    print(f"decoded : {morse_to_text(morse)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
