import operator

from enum import IntEnum
from typing import Callable
from functools import wraps

from ProtoTrainer.core.generic.base_context import ContextObject, FunctionObjectFactory, SortedList, BaseContext


class Priority(IntEnum):
    LOW = 0
    MEDIUM = 5
    HIGH = 10

class PriorityContextObject(ContextObject):
    """
    Function object class for wrapping regular functions with additional attributes.

    Attributes:
        priority (Priority): the execution priority of the function within the respective call stack.
        context (str): The container identifier this function belongs to.
    """
    def __init__(self, function: Callable, context: str, priority: Priority):
        super().__init__(function=function, context=context)
        self.priority = priority

    def get_priority(self):
        return self.priority

class PriorityContextType(FunctionObjectFactory):
    def __init__(self):
        super().__init__(PriorityContextObject)
    def __call__(self, function = None, *, priority: Priority = Priority.LOW) -> Callable:
        return super().__call__(function, context=self.name, priority=priority)


class PriorityList(SortedList[PriorityContextObject]):
    def __init__(self, iterable=None):
        super().__init__(iterable=iterable, ordering="descending")

    def __set_name__(self, owner, name):
        self.name = name

    def get_key(self, item: PriorityContextObject):
        return item.priority

    def cast_to_type(self, item) -> PriorityContextObject:
        if not isinstance(item, PriorityContextObject):
            return PriorityContextObject(item, context=self.name, priority=Priority.LOW)
        else:
            return item

class _PriorityContext(BaseContext):
    @classmethod
    def get_function_priority(cls,
                          func: PriorityContextObject,
                          raise_exception: bool = False) -> int | None:
        """
        Args:
            func: function that has been decorated with a context
            raise_exception: Whether to raise an exception when a context is not found. Otherwise,
                `None` is returned instead.

        Returns: the function rank

        Raises:
            ValueError: If a context is not found and ``raise_exception`` is True
        """
        try:
            prio = func.get_priority()
        except AttributeError:
            prio = None
        if prio is None and raise_exception:
            raise ValueError(f"No priority defined for this function")
        return prio


class Context(_PriorityContext):
    _CONTAINER_TYPE = PriorityList

    on_train_step_end: PriorityContextType
    on_epoch_step_end: PriorityContextType
    optimizer_post_step: PriorityContextType
    optimizer_pre_step: PriorityContextType
    on_eval_step_end: PriorityContextType
    on_eval_end: PriorityContextType

    def _insert(self, container: PriorityList, function: PriorityContextObject):
        container.append(function)


class ContextFunction:
    def __init__(self, function: Callable, condition=None, doc=None):
        self.function = function
        self.function_name = function.__name__
        self.to_decorate = {}
        self.condition = None

    def _deferred_decoration(self, function: Callable):
        self.to_decorate[function.__name__] = function

    def condition(self, func):
        prop = type(self)(self.function, func, self.__doc__)
        return prop


def execute_for_step(n: int | float, relate: str = "%"):
    """
    Wraps any callback function to be executed at step ``n`` according to the given condition, which is generated
    from ``relate``. If ``n`` is a float, the relative step will be computed according to the given context the function
    shall be executed at e.g. `0.5` in the epoch context results in `n = n * total_epochs` and `n = n * steps_per_epoch`
    for step based execution. ``n=-1`` executes the function on the last available epoch or step within an epoch.
    Functions that are decorated with `execute_for_step` but not decorated with any corresponding context won't
    be executed.

    Notes:
        Execution Table Examples:

        =========  =============   =========  ===================
        `n`        relate          context    executed
        =========  =============   =========  ===================
        ``5``         ``%``        step       every 5th step
        ``-1``        ``%``        step       every last step of each epoch
        ``0.5``       ``%``        step       twice per epoch
        ``0.5``       ``==``       step       only once at (`steps_per_epoch*0.5`)
        ``5``         ``>=``       epoch      every epoch starting at epoch 5
        ``-1``        ``%/==``     epoch      once at the last epoch
        ``0.5``       ``%``        epoch      twice during training
        ...
        =========  =============   =========  ===================

    Args:
        n: The number of steps to execute. Must be one of ``int[0,...,N]``, ``float[0.0,...,1.0]``, ``-1``.
        relate: The compare operator used to compute the condition at which the function will be executed.
         Can can be one of ``["%", "<=", "<", ">=", ">", "=="]``. "%" equates to "execute every `n` th step".

    Returns:
        The decorated function

    Raises:
        ValueError:
            If ``n`` is a float and not between [0,1].

            If ``n`` is an int and < 0 but != -1.

            If ``relate`` is not in ``["%", "<=", "<", ">=", ">", "=="]``.
    """

    def deferred_decoration(func):
        func.to_decorate = {"execute_for_step": {"kwargs": {"n": n, "relate": relate}}}
        return func

    if isinstance(n, int):
        if n < 0:
            if n == -1:
               return deferred_decoration
            else:
                raise ValueError(f"{n} is no valid value for n. Only '-1' is valid for negative integers")
        # postpone function decoration
    elif isinstance(n, float):
        if 0.0 < n <= 1.0:
            return deferred_decoration
        else:
            raise ValueError(f"n must be > 0.0 and < 1.0 but {n} was given")

    # else decorate directly
    op_dict = {
        "%": lambda x: operator.eq(operator.mod(x, n), 0),
        "<=": lambda x: operator.le(x, n),
        "<": lambda x: operator.lt(x, n),
        ">": lambda x: operator.gt(x, n),
        ">=": lambda x: operator.ge(x, n),
        "==": lambda x: operator.eq(x, n),
    }
    if relate not in op_dict.keys():
        raise ValueError(f"Relate must be one of {list(op_dict.keys())}")
    condition = op_dict[relate]

    def decorator_every_n_step(func):
        @wraps(func)
        def wrapper(self, current_step, *args, **kwargs):
            if condition(current_step):
                func(self, current_step, *args, **kwargs)

        return wrapper

    return decorator_every_n_step