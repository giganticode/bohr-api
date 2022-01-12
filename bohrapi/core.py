import functools
import logging
from abc import ABC
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from types import SimpleNamespace
from typing import Callable, Optional, Type, Dict, TypeVar, List

import pandas as pd
from bohrlabels.core import Labels, Label

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
    n_datapoints: Optional[int] = None


DataPointToLabelFunction = Callable


@dataclass(frozen=True)
class Task:
    name: str
    author: str
    description: Optional[str]
    top_artifact: ArtifactType
    labels: List[Label]
    training_dataset: Dataset
    test_datasets: Dict[Dataset, DataPointToLabelFunction]

    def __hash__(self):
        return hash(self.name)

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError(f'At least 2 labels have to be specified')
        if len(self.test_datasets) == 0:
            raise ValueError(f'At least 1 test dataset has to be specified')

        dataset_name_set = set()
        for dataset in self.datasets:
            count_before = len(dataset_name_set)
            dataset_name_set.add(dataset.id)
            count_after = len(dataset_name_set)
            if count_after == count_before:
                raise ValueError(f"Dataset {dataset.id} is present more than once.")

    @property
    def datasets(self) -> List[Dataset]:
        return [self.training_dataset] + list(self.test_datasets.keys())

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f'Unknown dataset: {dataset_id}')


@dataclass(frozen=True)
class Experiment:
    name: str
    task: Task
    heuristics_classifier: str

    def __post_init__(self):
        if self.heuristic_groups is not None and len(self.heuristic_groups) == 0:
            raise ValueError(f'At least 1 heuristic group has to be specified')

    @property
    def heuristic_groups(self) -> Optional[List[str]]:
        if '@' not in self.heuristics_classifier:
            return None

        paths, revision = self.heuristics_classifier.split('@')
        return paths.split(':')

    @property
    def revision(self) -> str:
        return self.heuristics_classifier.split('@')[-1]


@dataclass(frozen=True)
class Workspace:
    bohr_runtime_version: str
    experiments: List[Experiment]

    def get_experiment_by_name(self, exp_name: str) -> Experiment:
        for exp in self.experiments:
            if exp.name == exp_name:
                return exp
        raise ValueError(f'Unknown experiment: {exp_name}')

    def get_task_by_name(self, task_name: str) -> Task:
        for exp in self.experiments:
            if exp.task.name == task_name:
                return exp.task
        raise ValueError(f'Unknown task: {task_name}')

    def get_dataset_by_id(self, id: str) -> Dataset:
        for run in self.experiments:
            try:
                return run.task.get_dataset_by_id(id)
            except ValueError:
                pass
        raise ValueError(f'Unknown dataset: {id}')
