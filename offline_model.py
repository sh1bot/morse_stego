"""
offline_model -- a tiny GPT-2 trained offline, for when Hugging Face is out of
reach. morse_stego uses this only as a fallback; keeping it here keeps the core
file tight.

The model is deliberately small and the corpus is a "word-ending pangram": short
words grouped so that every final-letter *class* -- vowel (dot), consonant
(dash), and 'y' (gap) -- follows every other, plus a '.' sentence-ender. That is
all the constrained search needs, since a morse symbol is read off a word's last
letter. A few hundred training steps concentrate the next-token distribution
enough that the best-first search actually finds legal cover text (an untrained
random model stays ~uniform, which defeats the search no matter how big its
vocabulary). The trained weights are cached on disk, so the ~15s train happens
once and later runs just load.
"""

import os

# Short words grouped by the class of their final letter. Balanced coverage of
# all three morse symbols is what makes arbitrary messages encodable.
_VOWEL = ["be", "we", "he", "me", "she", "see", "sea", "tea", "the", "so", "go",
          "no", "to", "do", "two", "blue", "true", "free", "tree", "toe", "a"]
_CONS = ["is", "it", "in", "on", "an", "at", "of", "or", "as", "us", "sun",
         "run", "man", "can", "red", "bed", "dog", "cat", "top", "big", "far",
         "for", "war", "not", "let"]
_YEND = ["by", "my", "day", "way", "say", "may", "sky", "dry", "try", "fly",
         "cry", "why", "any", "many", "very", "stay", "play", "gray"]

# Cache dir + a version tag so changing the corpus or hyperparameters retrains.
CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "morse_stego", "offline-gpt2-v2")

_STEPS = 900
_SEED = 0

# Mirrors morse_stego's default --prompt. Seeding the corpus with it lets the
# tiny model actually continue the default prompt (otherwise it is out of
# distribution and its continuations are noise, so the search dead-ends).
_PROMPT = "The weather today is"


def _corpus():
    """A deterministic corpus of short sentences cycling the ending classes so
    every word-ending kind is seen following every other, half of them seeded
    with the default prompt so the model learns to continue it."""
    orders = [(_VOWEL, _CONS, _YEND), (_CONS, _VOWEL, _YEND), (_YEND, _VOWEL, _CONS),
              (_VOWEL, _YEND, _CONS), (_CONS, _YEND, _VOWEL), (_YEND, _CONS, _VOWEL)]
    sents = []
    for a, b, c in orders:
        for i in range(30):
            body = (f"{a[i % len(a)]} {b[i % len(b)]} {c[i % len(c)]} "
                    f"{a[(i + 1) % len(a)]} {b[(i + 2) % len(b)]}")
            sents.append(f"{body} .")
            sents.append(f"{_PROMPT} {body} .")
    return " ".join(sents)


def _build_tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast
    inner = ByteLevelBPETokenizer()
    inner.train_from_iterator([_corpus() * 3], vocab_size=1000, min_frequency=1,
                              special_tokens=["<|endoftext|>"])
    tok = GPT2TokenizerFast(tokenizer_object=inner._tokenizer)
    tok.add_special_tokens({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"})
    eos = inner.token_to_id("<|endoftext|>")
    tok.eos_token_id = tok.pad_token_id = tok.bos_token_id = eos
    return tok


def _train(tok, steps):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(_SEED)
    cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=1024,
                     n_embd=128, n_layer=3, n_head=4,
                     eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)
    model = GPT2LMHeadModel(cfg)
    model.train()
    ids = tok(_corpus() * 3, return_tensors="pt")["input_ids"][0]
    window = 96
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(steps):
        i = (step * window) % max(1, len(ids) - window - 1)
        batch = ids[i:i + window].unsqueeze(0)
        loss = model(batch, labels=batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model


def _finish(tok, model):
    """Align the eos/pad ids to the small vocab and put the model in eval mode."""
    model.config.eos_token_id = model.config.bos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.eos_token_id
    model.eval()
    return tok, model


def load_offline_model():
    """Return (tokenizer, model) for the offline tiny GPT-2, training and caching
    it on first use and loading the cached weights thereafter."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    if os.path.isdir(CACHE_DIR):
        tok = GPT2TokenizerFast.from_pretrained(CACHE_DIR)
        return _finish(tok, GPT2LMHeadModel.from_pretrained(CACHE_DIR))

    print(f"[training offline tiny GPT-2 ({_STEPS} steps); cached to {CACHE_DIR}]")
    tok = _build_tokenizer()
    model = _train(tok, _STEPS)
    tok, model = _finish(tok, model)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tok.save_pretrained(CACHE_DIR)
        model.save_pretrained(CACHE_DIR)
    except OSError as e:                          # cache is a nicety, not required
        print(f"[could not cache offline model: {e}]")
    return tok, model
