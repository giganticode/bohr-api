from dataclasses import dataclass
from functools import cached_property
from typing import List, Set

from bohrapi.core import Artifact
from bohrapi.util.misc import NgramSet


@dataclass
class Issue(Artifact):

    @property
    def title(self):
        return self.raw_data['title']

    @property
    def body(self):
        return self.raw_data['body']

    @property
    def labels(self) -> List[str]:
        """
        >>> from types import SimpleNamespace
        >>> issue = Issue({})
        >>> issue.labels
        ''
        """
        if 'labels' not in self.raw_data or self.raw_data['labels'] is None:
            return []
        return list(map(lambda s: s.strip().lower(), self.raw_data['labels'].split(',')))

    @cached_property
    def stemmed_labels(self) -> Set[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return {stemmer.stem(label) for label in self.labels}

    @cached_property
    def tokens(self) -> Set[str]:
        from bohrapi.util.nlp import safe_tokenize

        if self.body is None:
            return set()
        return safe_tokenize(self.body)

    @cached_property
    def ordered_stems(self) -> List[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in self.tokens]

    @cached_property
    def stemmed_ngrams(self) -> NgramSet:
        from nltk import bigrams

        return set(self.ordered_stems).union(set(bigrams(self.ordered_stems)))
