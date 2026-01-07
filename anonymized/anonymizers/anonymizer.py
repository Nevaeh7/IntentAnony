from abc import ABC, abstractmethod
from typing import List, Iterator, Dict, Any


class Anonymizer(ABC):
    @abstractmethod
    def anonymize(self, text: str) -> str:
        pass

    @abstractmethod
    def anonymize_profiles(self, profiles: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        pass
