from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

class ScribeEnv(ABC):

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        ...

    @abstractmethod
    def get_task_description(self) -> str:
        ...

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        ...

    def render(self) -> Optional[str]:
        return None

    def close(self) -> None:
        pass

class ThreadPoolScribeMultiEnv:

    def __init__(self, envs: List[ScribeEnv], max_workers: Optional[int] = None):
        self.envs = envs
        self.num_processes = len(envs)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or self.num_processes
        )

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if seed is None or isinstance(seed, int):
            seeds = [seed] * self.num_processes
        else:
            seeds = list(seed)
            assert len(seeds) == self.num_processes, (
                f"seed length {len(seeds)} != num_processes {self.num_processes}"
            )

        if options is None:
            options = [None] * self.num_processes
        else:
            assert len(options) == self.num_processes, (
                f"options length {len(options)} != num_processes {self.num_processes}"
            )

        futures = [
            self._executor.submit(env.reset, s, opt)
            for env, s, opt in zip(self.envs, seeds, options)
        ]
        results = [f.result() for f in futures]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(
        self,
        actions: List[Any],
    ) -> Tuple[List[str], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        assert len(actions) == self.num_processes, (
            f"actions length {len(actions)} != num_processes {self.num_processes}"
        )

        futures = [
            self._executor.submit(env.step, action)
            for env, action in zip(self.envs, actions)
        ]
        results = [f.result() for f in futures]
        obs_list, reward_list, terminated_list, truncated_list, info_list = zip(*results)
        return (
            list(obs_list),
            list(reward_list),
            list(terminated_list),
            list(truncated_list),
            list(info_list),
        )

    def get_task_descriptions(self) -> List[str]:
        return [env.get_task_description() for env in self.envs]

    def get_tools(self) -> List[List[Dict[str, Any]]]:
        return [env.get_tools() for env in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)

__all__ = ["ScribeEnv", "ThreadPoolScribeMultiEnv"]
