
import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Callable, Optional, Type, Dict, TypeVar, List, Set, Tuple, Union

from frozendict import frozendict

from bohrlabels.core import Labels, Label, NumericLabel, LabelSubclass
from bohrlabels.labels import CommitLabel

logger = logging.getLogger(__name__)

HeuristicFunction = Callable[..., Optional[Labels]]


@dataclass(repr=False)
class Artifact:
    raw_data: Dict

    def __repr__(self):
        return super(Artifact, self).__repr__()


@dataclass(repr=False)
class MergeableArtifact(Artifact, ABC):
    @abstractmethod
    def single_identity(self) -> str:
        pass


ArtifactSubclass = TypeVar("ArtifactSubclass", bound="Artifact")
ArtifactType = Type[ArtifactSubclass]


class HeuristicObj:
    def __init__(
        self, func: Callable, non_safe_func: Callable, artifact_type_applied_to: ArtifactType, resources=None
    ):
        self.artifact_type_applied_to = artifact_type_applied_to
        self.resources = resources
        self.func = func
        self.non_safe_func = non_safe_func
        functools.update_wrapper(self, func)

    def __call__(self, artifact: Artifact, *args, **kwargs) -> Label:
        return self.func(artifact, *args, **kwargs)


class Heuristic:
    def __init__(self,
                 artifact_type_applied_to: Type[Artifact],
                 artifact_type_applied_to2: Optional[Type[Artifact]] = None):
        self.artifact_type_applied_to = artifact_type_applied_to
        self.artifact_type_applied_to2 = artifact_type_applied_to2

    def check_artifact_types(self, artifact: Union[Artifact, Tuple[Artifact, Artifact]], name) -> bool:
        if self.artifact_type_applied_to2 is None:
            if isinstance(artifact, tuple):
                raise TypeError(f"Expected artifact of type {self.artifact_type_applied_to.__name__}, got tuple")
            if not isinstance(artifact, self.artifact_type_applied_to):
                raise TypeError(f"Heuristic {name} can only be applied to {self.artifact_type_applied_to.__name__} object, not int")
        else:
            if not isinstance(artifact, tuple):
                raise TypeError(f'Heuristic {name} accepts only tuple of two artifacts')
            if not (self.artifact_type_applied_to == artifact[0].__class__ and self.artifact_type_applied_to2 == artifact[1].__class__):
                raise TypeError(f'Heuristic {name} can only be applied to {self.artifact_type_applied_to.__name__} and {self.artifact_type_applied_to2.__name__}')

    def get_artifact_safe_func(self, f: HeuristicFunction) -> HeuristicFunction:
        def func(artifact: Union[Artifact, Tuple[Artifact, Artifact]], *args, **kwargs):
            self.check_artifact_types(artifact, name=f.__name__)
            try:
                return f(artifact, *args, **kwargs)
            except (
                    ValueError,
                    KeyError,
                    AttributeError,
                    IndexError,
                    TypeError,
                    CalledProcessError,
            ) as ex:
                logger.exception(
                    "Exception thrown while applying heuristic, "
                    "skipping the heuristic for this datapoint ..."
                )
                raise ex
                # return None

        return functools.wraps(f)(func)

    def __call__(self, f: Callable[..., Label]) -> HeuristicObj:
        safe_func = self.get_artifact_safe_func(f)
        return HeuristicObj(safe_func, f, self.artifact_type_applied_to)


@dataclass(frozen=True)
class Dataset(ABC):
    id: str
    top_artifact: ArtifactType
    query: Optional[Dict] = field(compare=False, default=None)
    projection: Optional[Dict] = field(compare=False, default=None)
    n_datapoints: Optional[int] = None


@dataclass(frozen=True)
class MatchingDataset(Dataset):
    pass


DataPointToLabelFunction = Callable


@dataclass(frozen=True)
class Task:
    name: str
    author: str
    description: Optional[str]
    top_artifact: ArtifactType
    labels: List[Union[Label, NumericLabel]] = field()
    test_datasets: Dict[Dataset, DataPointToLabelFunction]
    hierarchy: Optional[Type[LabelSubclass]] = None


@dataclass(frozen=True)
class Experiment:
    name: str
    task: Task
    train_dataset: Dataset
    class_balance: Optional[Tuple[float, ...]] = None
    heuristics_classifier: Optional[str] = None
    extra_test_datasets: Dict[Dataset, Callable] = field(default_factory=frozendict)


@dataclass(frozen=True)
class Workspace:
    bohr_runtime_version: str
    experiments: List[Experiment]
