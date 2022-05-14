from bohrapi.artifacts.commit import Commit
from bohrapi.artifacts.commit_file import CommitFile
from bohrapi.artifacts.commit_message import CommitMessage
from bohrapi.artifacts.issue import Issue
from bohrapi.artifacts.method import Method
from bohrapi.artifacts.identity import Identity
from bohrlabels.core import Label

artifact_map = {
    "commit": Commit,
    "commit_file": CommitFile,
    "commit_message": CommitMessage,
    "issue": Issue,
    "method": Method,
    "identity": Identity,
    "label": Label,
}
