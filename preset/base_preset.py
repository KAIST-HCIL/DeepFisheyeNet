from abc import ABC, abstractmethod

class BasePreset(ABC):
    """ Preset is kind of a recipe.
        It has a set of options that should be predefined.
        Preset overrides default options and arguments.
    """
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def modify_options(cls, opt):
        pass
