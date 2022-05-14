import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Set

from bohrapi.artifacts.commit_file import CommitFile
from bohrapi.artifacts.commit_message import CommitMessage
from bohrapi.artifacts.issue import Issue
from bohrapi.core import Artifact
from bohrapi.util.misc import NgramSet

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class Commit(Artifact):

    @staticmethod
    def important_fields_map():
        return {'_id': 'sha', 'message': 'message'}

    @property
    def sha(self):
        return self.raw_data['_id']

    @property
    def owner(self):
        return self.raw_data['owner']

    @property
    def repository(self):
        return self.raw_data['repository']

    @cached_property
    def message(self):
        return CommitMessage(self.raw_data['message'])

    @cached_property
    def clean_message(self):
        if 'message' not in self.raw_data or self.raw_data['message'] is None:
            CommitMessage(None)
        from bohrapi.util.messagecleaner import clean_message
        cleaned_message = clean_message(str(self.raw_data['message'])).clean_message
        return CommitMessage(cleaned_message)

    @cached_property
    def issues(self) -> List[Issue]:
        return list(map(lambda x: Issue(x), self.raw_data['issues'])) if "issues" in self.raw_data else []

    @cached_property
    def commit_files(self) -> List[CommitFile]:
        return list(map(lambda x: CommitFile(x), self.raw_data["files"])) if "files" in self.raw_data else []

    def __hash__(self):
        return hash(self.sha)

    def issues_match_label(self, stemmed_labels: Set[str]) -> bool:
        for issue in self.issues:
            if not issue.stemmed_labels.isdisjoint(stemmed_labels):
                return True
        return False

    def issues_match_ngrams(self, stemmed_keywords: NgramSet) -> bool:
        for issue in self.issues:
            if not issue.stemmed_ngrams.isdisjoint(stemmed_keywords):
                return True
        return False

    def __repr__(self):
        return super(Commit, self).__repr__()
