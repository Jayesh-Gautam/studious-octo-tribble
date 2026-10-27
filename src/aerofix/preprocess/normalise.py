"""Text normalisation pipeline: tokenise -> abbreviations -> spelling -> stem | lemma.

    norm = Normaliser.fit(train_texts, cfg)      # learns vocab, LM, spell dictionary
    norm.normalise("RPLCD L/H VLV, CORRISION FOUND", morph="lemma")
    -> "REMOVE ..."  (see tests/test_normalise.py for exact examples)

Stemming vs lemmatisation (NLTK):
  * Porter stemmer chops suffixes by rule: CRACKED -> CRACK, CORROSION -> CORROS.
    Fast, but produces non-words and can merge unrelated words.
  * WordNet lemmatiser maps to a dictionary form using the part of speech:
    REPLACED (verb) -> REPLACE, LEAKS (noun) -> LEAK. Slower (needs a POS tagger),
    but output stays readable, which matters when we show text in the app.
Which one helps more is measured, not assumed (reports/normalisation.md).
"""
from __future__ import annotations

import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

from aerofix.preprocess.abbrev import expand, load_abbrev
from aerofix.preprocess.ngram_lm import NGramLM
from aerofix.preprocess.spell import SpellCorrector
from aerofix.preprocess.tokenize import tokenize

_WN_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}


def _plain(tok: str) -> bool:
    """Only plain words are stemmed / lemmatised; [REF], P/N, 737-700 stay as they are."""
    return tok.isalpha()


class Normaliser:
    def __init__(self, abbrev: dict[str, list[str]], spell: SpellCorrector, lm: NGramLM,
                 stopwords: set[str] = frozenset()):
        self.abbrev, self.spell, self.lm, self.stopwords = abbrev, spell, lm, set(stopwords)
        self._stemmer = self._lemmatizer = self._tagger = None

    @classmethod
    def fit(cls, train_texts, cfg: dict) -> "Normaliser":
        """Fit on TRAIN text only (masked text, so the LM never sees reference numbers)."""
        ncfg = cfg["normalise"]
        abbrev = load_abbrev(cfg["project_root"] / ncfg["abbrev_path"])
        docs = [expand(tokenize(t), abbrev) for t in train_texts]
        counts = Counter(t for d in docs for t in d)
        lm = NGramLM(order=ncfg["lm"]["order"], smoothing="kn", discount=ncfg["lm"]["discount"],
                     unk_min_count=ncfg["lm"]["unk_min_count"]).fit(docs)
        s = ncfg["spell"]
        protected = set(abbrev) | {w for exp in abbrev.values() for w in exp}
        spell = SpellCorrector(counts, lm, protected, s["dict_min_count"], s["max_edit"],
                               s["min_len"], s["edit_penalty"])
        stop = set()
        if ncfg["remove_stopwords"]:
            from nltk.corpus import stopwords
            stop = {w.upper() for w in stopwords.words("english")} - set(ncfg["keep_stopwords"])
        return cls(abbrev, spell, lm, stop)

    # ---------- steps ----------
    def tokens(self, text: str, abbrev: bool = True, spell: bool = True, morph: str = "none") -> list[str]:
        toks = tokenize(text)
        if abbrev:
            toks = expand(toks, self.abbrev)
        if spell:
            toks = self.spell.correct(toks)
        if self.stopwords:
            toks = [t for t in toks if t not in self.stopwords]
        if morph == "stem":
            toks = self.stem(toks)
        elif morph == "lemma":
            toks = self.lemmatise(toks)
        return toks

    def normalise(self, text: str, abbrev: bool = True, spell: bool = True, morph: str = "none") -> str:
        return " ".join(self.tokens(text, abbrev, spell, morph))

    def stem(self, toks: list[str]) -> list[str]:
        if self._stemmer is None:
            from nltk.stem import PorterStemmer
            self._stemmer = lru_cache(maxsize=100_000)(PorterStemmer().stem)
        return [self._stemmer(t.lower()).upper() if _plain(t) else t for t in toks]

    def lemmatise(self, toks: list[str]) -> list[str]:
        if self._lemmatizer is None:
            from nltk.stem import WordNetLemmatizer
            from nltk.tag.perceptron import PerceptronTagger
            self._lemmatizer = lru_cache(maxsize=200_000)(WordNetLemmatizer().lemmatize)
            self._tagger = PerceptronTagger()  # one instance: nltk.pos_tag reloads it every call
        tagged = self._tagger.tag([t.lower() for t in toks])
        return [self._lemmatizer(w, _WN_POS.get(tag[0], "n")).upper() if _plain(t) else t
                for t, (w, tag) in zip(toks, tagged)]

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        self._stemmer = self._lemmatizer = self._tagger = None  # caches/tagger are rebuilt on use
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Normaliser":
        with open(path, "rb") as f:
            return pickle.load(f)
