from types import MethodType
from typing import Union, Type, TypeVar, Literal, Any, Callable
from functools import partial, update_wrapper
from collections.abc import MutableSequence, Iterable


#_C = TypeVar('_C', bound=Union[list, dict, set])
_C = TypeVar('_C', bound=Iterable[Callable])
#_C_Type = Type[list | dict | set]

class _FunctionObject:
    def __init__(self, function: Callable):
        self._function = function
        self._partial = False
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    # we need one level of indirection for a partial call since we can't modify __call__
    # and don't want to decorate the original function
    def call(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    def __get__(self, instance, owner):
        if not self._partial:
            self.call = partial(self.call, instance)
            self._partial = True
        return self

    def __repr__(self):
        return f"{super().__repr__()} wrapping {self._function}"


class ContextObject(_FunctionObject):
    """
    Attributes:
        context (str): The container identifier this function belongs to.
    """
    def __init__(self, function: Callable, context: str):
        super().__init__(function)
        self.context = context

    def get_context(self):
        return self.context

class FunctionObjectFactory:
    """
    Factory class generating instances of FunctionObject decorators.
    """
    def __init__(self, cls_type: type):
        self.cls_type = cls_type

    def __set_name__(self, owner, name):
        self.name = name

    def __call__(self, function = None, **kwargs) -> Callable:
        if function is None:
            # call the base class function to propagate **kwargs without conflict
            return partial(MethodType(FunctionObjectFactory.__call__, self), **kwargs)
        return self.cls_type(function, **kwargs)

def dec(cls_type):
    def wrapper(function = None, **kwargs):
        if function is None:
            return partial(wrapper, **kwargs)
        return cls_type(function, **kwargs)

T = TypeVar('T')
class SortedList(MutableSequence[T]):
    """
    Self-sorting list. List is sorted before every read access if the values have changed.
    Objects in the list must expose a common interface accessed via `get_key()`. Items that
    are not exposing an interface during insertion can be cast to the appropriate type by overwriting
    `cast_to_type()`. Otherwise, `cast_to_type()` is just the identity function.
    """
    def __init__(self, iterable=None, *, ordering: Literal["ascending", "descending"] = "descending"):
        self._data: list[T] = list() if iterable is None else list(iterable)
        self.reverse = True if ordering == "descending" else False
        self._values_changed = False

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if self._values_changed:
            self.sort()
        return self._data[i]

    def __iter__(self):
        if self._values_changed:
            self.sort()
        return iter(self._data)

    def __delitem__(self, i):
        self._values_changed = True
        del self._data[i]

    def __setitem__(self, i, value):
        self._values_changed = True
        self._data[i] = self.cast_to_type(value)

    def insert(self, i, value):
        self._values_changed = True
        self._data.insert(i, self.cast_to_type(value))

    def get_key(self, item) -> Any:
        raise NotImplementedError

    def cast_to_type(self, item) -> T:
        return item

    def sort(self, *, reverse: bool = None):
        """Return the sorted list. Overwrites the ordering if reverse is not ``None``"""
        if reverse is not None and reverse != self.reverse:
            self.reverse = reverse
        self._data.sort(key=self.get_key, reverse=self.reverse)
        self._values_changed = False

class _ContextMeta(type):
    """
    Metaclass to enable dataclass like behavior. Class attributes of type FunctionObjectFactory are initialized
    automatically, depending on the type of factory and initialization list provided in the concrete class.
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # annotated_types = list(get_type_hints(cls).values())
        # search for ContextType annotations, __annotations__ is fixed to the class scope.
        # get_type_hints() includes subtype annotations
        if "__annotations__" in attrs:
            context_type_attributes = {name: cls_type for name, cls_type in attrs['__annotations__'].items() if
                                       isinstance(cls_type, type) and issubclass(cls_type, FunctionObjectFactory)} # short circuit
            if len(context_type_attributes) > 0:
                _tmp_cls_list = list(context_type_attributes.values())
                first_type = _tmp_cls_list[0]
                # restriction: all ContextTypes must be of same type (might be a design choice)
                diverging_types = all([first_type is not other_type for other_type in _tmp_cls_list])
                if diverging_types:
                    raise RuntimeError("Class Attributes of subtypes of 'BaseContextType' must be of same type.")
                # now init the cls attributes, search for init list
                init_key = "_INIT_LIST"
                init_list = attrs.get(init_key) if init_key in attrs else getattr(bases[0], init_key)
                for attr_name, attr_type in context_type_attributes.items():
                    # set_name is only called in cls.__new__, so we missed timing and have to call manually
                    cls_obj = attr_type(*init_list) if init_list else attr_type()
                    setattr(cls, attr_name, cls_obj)
                    getattr(cls, attr_name).__set_name__(cls, attr_name)

def _run(self: Iterable, *args, **kwargs):
    for func in self:
        func(*args, **kwargs)

class BaseContext(metaclass=_ContextMeta):
    """
    Base container class hosting functions that are annotated with a context. Each context attribute is a decorator of
    the desired type, providing a corresponding container on an instance level.
    Context classes are function managers that delegate functions to a container during runtime, useful if
    functions of the same interface need to be grouped and executed at once. Can also be used as decorators, adding
    attributes and helper functions to the decorated functions.

    Examples::

        # define a custom decorator factory, returning ContextObject instances
        class ContextObjectType(FunctionObjectFactory):
            def __init__(self):
                super().__init__(ContextObject)

            # the decorator wrapping functions to a ContextObject.
            def __call__(self, function):
                # everything in super()__call__ is passed to ContextObject.__init__
                super().__call__(function, name=self.name)

        class MyContext(BaseContext):
            _CONTAINER_TYPE = list

            container1: ContextObjectType
            container2: ContextObjectType

        class A:
            def __init__(self):
                self.context_manager = MyContext()
                self.context_manager.register_all_functions()

            @MyContext.container1
            def func1(self):
                pass

            @MyContext.container2
            def func2(self):
                pass

            def do_something(self):
                self.context_manager.container1.run()


    Attributes:
        _CONTAINER_TYPE (type): the container class to host the functions. Must be exposing a callable when called with
            `next(iter(container))`
        _INIT_LIST (list): the list to initialize FunctionObjectFactory instances and child classes. Defaults to None.
    """

    _CONTAINER_TYPE: type[Iterable[Callable]]
    _INIT_LIST: list | None = None

    def __init__(self):
        current_class = type(self)
        context_type_attributes = {name for name, attr in vars(current_class).items() if
                                   isinstance(attr, FunctionObjectFactory)}
        if not len(context_type_attributes):
            raise RuntimeError(f"No class attributes defined for {self.__class__.__name__}")
        # attach run function to the container
        wrapped_cls = type(current_class._CONTAINER_TYPE.__name__, (current_class._CONTAINER_TYPE,), {'run': _run})
        for attr_name in context_type_attributes:
            # set the container
            setattr(self, attr_name, wrapped_cls())
            getattr(self, attr_name).__set_name__(self, attr_name)

    def _insert(self, container: _C, function: ContextObject):
        raise NotImplementedError

    def register(self, function: ContextObject):
        try:
            function_context = function.get_context()
        except AttributeError:
            raise ValueError("Trying to register a raw function: function must be wrapped in a Context object")
        if function_context is None:
            raise ValueError("Trying to register a function with no specified context. "
                             "The function you are trying to bind must be decorated with a context subtype")
        context_container: _C = getattr(self, function_context, None)
        if context_container is None:
            raise RuntimeError(f"No container defined for the context {function_context}")
        self._insert(context_container, function)

    def register_all_functions(self, owner):
        functions = self.retrieve_decorated_functions(owner)
        for function in functions:
            self.register(getattr(owner, function))

    @classmethod
    def retrieve_decorated_functions(cls, owner: object) -> list[str]:
        """Retrieve all functions within owner, that have been decorated with the specific context"""

        def condition(func_name: str):
            _func_body = getattr(owner, func_name)
            return (not func_name.startswith("__") and callable(_func_body) and
                    isinstance(_func_body, ContextObject))

        return [func for func in dir(owner) if condition(func)]

    @classmethod
    def get_function_context(cls,
                             func: ContextObject,
                             raise_exception: bool = False
                             ) -> str | None:
        """
        Args:
            func: function that has been decorated with a context
            raise_exception: Whether to raise an exception when a context is not found. Otherwise,
                `None` is returned instead.

        Returns: the context identifier

        Raises:
            ValueError: If a context is not found and ``raise_exception`` is True

        """
        try:
            context = func.get_context()
        except AttributeError:
            context = None
        if context is None and raise_exception:
            raise ValueError(f"No context defined for function {func}")
        return context

