from enum import Enum, EnumMeta
from torch.optim import Adam

# TODO: linter complains

class BaseEnumMeta(EnumMeta):

    def __contains__(cls, other):
        try:
            cls(other)
        except ValueError:
            return False
        else:
            return True

        #return other in cls.__members__.keys()

    def __iter__(cls):
        # hide all private callable members _name of this enum
        return (cls._member_map_[name] for name in cls._member_names_ if "_" not in name)

    @property
    def choices(cls):
        return [m.value for m in cls]

class BaseEnum(Enum, metaclass=BaseEnumMeta):
    ...


class CallableEnum(Enum, metaclass=BaseEnumMeta):
    def __call__(self, *args, **kwargs):
        try:
            return self.__class__["_" + self.name].value
        except KeyError:
            raise NotImplementedError("No callable defined for this enum member")


class AverageModelType(BaseEnum):
    """
    Supported model averaging types
    Attributes:
        SWE: Stochastic Weight Averaging
        EMA: Exponential Moving Average

    """
    SWA = "swa"
    EMA = "ema"

from typing import Callable
class OptimizerType(CallableEnum):
    """
    Supported optimizer types
    Attributes:
        ADAM: Adam Optimizer
        ADAMW: Adaptive Moving Average Weighting Optimizer
        RMSPROP: RMSprop Optimizer
        SGD: SGD Optimizer
    """
    ADAM= "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    SGD = "sgd"

    _ADAM = Adam


