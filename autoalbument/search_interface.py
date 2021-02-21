from abc import ABC, abstractmethod


class SearcherBase(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def search(self):
        ...
