from dataclasses import dataclass
from functools import cached_property
from typing import List, Set, Optional

from bohrapi.core import Artifact
from bohrapi.util.misc import NgramSet


@dataclass
class CommitMessage(Artifact):

    @property
    def raw(self) -> str:
        return str(self.raw_data)

    @cached_property
    def tokens(self) -> Set[str]:
        from bohrapi.util.nlp import safe_tokenize

        if self.raw_data is None:
            return set()
        return safe_tokenize(self.raw_data)

    @cached_property
    def ordered_stems(self) -> List[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in self.tokens]

    @cached_property
    def stemmed_ngrams(self) -> NgramSet:
        from nltk import bigrams

        return set(self.ordered_stems).union(set(bigrams(self.ordered_stems)))

    def match_ngrams(self, stemmed_keywords: NgramSet) -> bool:
        return not self.stemmed_ngrams.isdisjoint(stemmed_keywords)
