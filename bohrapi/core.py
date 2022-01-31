import functools
import logging
from abc import ABC
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Callable, Optional, Type, Dict, TypeVar, List, Set, Tuple, Union

from frozendict import frozendict

from bohrlabels.core import Labels, Label, NumericLabel, LabelSubclass, to_numeric_label

logger = logging.getLogger(__name__)

HeuristicFunction = Callable[..., Optional[Labels]]


@dataclass(repr=False)
class Artifact:
    raw_data: Dict

    def __repr__(self):
        return super(Artifact, self).__repr__()


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
    def __init__(self, artifact_type_applied_to: Type[Artifact]):
        self.artifact_type_applied_to = artifact_type_applied_to

    def get_artifact_safe_func(self, f: HeuristicFunction) -> HeuristicFunction:
        def func(artifact, *args, **kwargs):
            if not isinstance(artifact, self.artifact_type_applied_to):
                raise ValueError("Not right artifact")
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

    def __lt__(self, other):
        if not isinstance(other, Dataset):
            raise ValueError(f'Cannot compare {Dataset.__name__} with {type(other).__name__}')

        return self.id < other.id


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

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def infer_hierarchy(labels: List[Union[Label, NumericLabel]], hierarchy: Optional[Type[LabelSubclass]]) -> Type[LabelSubclass]:
        """
        >>> from bohrlabels.labels import CommitLabel, SStuB
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], None)
        <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], CommitLabel)
        <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], SStuB)
        Traceback (most recent call last):
        ...
        ValueError: Passed hierarchy is: <enum 'SStuB'>, and one of the categories is <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([SStuB.WrongFunction, CommitLabel.Feature], None)
        Traceback (most recent call last):
        ...
        ValueError: Cannot specify categories from different hierarchies: <enum 'CommitLabel'> and <enum 'SStuB'>
        """
        inferred_hierarchy = hierarchy
        for label in labels:
            if isinstance(label, Label):
                label = label.to_numeric_label()

            if isinstance(label, NumericLabel):
                if inferred_hierarchy is None:
                    inferred_hierarchy = label.hierarchy
                elif label.hierarchy != inferred_hierarchy:
                    if hierarchy is None:
                        raise ValueError(
                            f"Cannot specify categories from different hierarchies: {label.hierarchy} and {inferred_hierarchy}"
                        )
                    else:
                        raise ValueError(
                            f"Passed hierarchy is: {inferred_hierarchy}, and one of the categories is {label.hierarchy}"
                        )
            elif isinstance(label, int):
                pass
            else:
                raise AssertionError()
        if inferred_hierarchy is None:
            raise ValueError('Cannot infer which hierarchy to use. Please pass `hierarchy` argument')

        return inferred_hierarchy

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError(f'At least 2 labels have to be specified')
        hierarchy = self.infer_hierarchy(self.labels, self.hierarchy)
        numeric_labels = [to_numeric_label(label, hierarchy) for label in self.labels]
        object.__setattr__(self, 'labels', numeric_labels)
        object.__setattr__(self, 'hierarchy', hierarchy)
        if len(self.test_datasets) == 0:
            raise ValueError(f'At least 1 test dataset has to be specified')

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.get_test_datasets():
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f"Dataset {dataset_id} is not found in task {self.name}")

    def get_test_datasets(self) -> List[Dataset]:
        return list(self.test_datasets.keys())


@dataclass(frozen=True)
class Experiment:
    name: str
    task: Task
    train_dataset: Dataset
    heuristics_classifier: str
    extra_test_datasets: Dict[Dataset, Callable] = field(default_factory=frozendict)

    def __post_init__(self):
        if self.heuristic_groups is not None and len(self.heuristic_groups) == 0:
            raise ValueError(f'At least 1 heuristic group has to be specified')

        dataset_name_set = set()
        for dataset in self.datasets:
            count_before = len(dataset_name_set)
            dataset_name_set.add(dataset.id)
            count_after = len(dataset_name_set)
            if count_after == count_before:
                raise ValueError(f"Dataset {dataset.id} is present more than once.")

    @property
    def heuristic_groups(self) -> Optional[List[str]]:
        if '@' not in self.heuristics_classifier:
            return None

        paths, revision = self.heuristics_classifier.split('@')
        return paths.split(':')

    @property
    def revision(self) -> str:
        return self.heuristics_classifier.split('@')[-1]

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f'Unknown dataset: {dataset_id}')

    @property
    def datasets(self) -> List[Dataset]:
        return self.task.get_test_datasets() + [self.train_dataset] + list(self.extra_test_datasets.keys())


@dataclass(frozen=True)
class Workspace:
    bohr_runtime_version: str
    experiments: List[Experiment]

    def get_experiment_by_name(self, exp_name: str) -> Experiment:
        for exp in self.experiments:
            if exp.name == exp_name:
                return exp
        raise ValueError(f'Unknown experiment: {exp_name}, possible values are {list(map(lambda e: e.name, self.experiments))}')

    def get_task_by_name(self, task_name: str) -> Task:
        for exp in self.experiments:
            if exp.task.name == task_name:
                return exp.task
        raise ValueError(f'Unknown task: {task_name}')

    def get_dataset_by_id(self, id: str) -> Dataset:
        for exp in self.experiments:
            try:
                return exp.get_dataset_by_id(id)
            except ValueError:
                pass
        raise ValueError(f'Unknown dataset: {id}')
