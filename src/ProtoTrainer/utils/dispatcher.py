import types
import weakref
from abc import get_cache_token
from enum import Enum, EnumMeta
from typing import get_type_hints
from functools import update_wrapper


def enumdispatch(func):
    """Single-dispatch generic function decorator for enum members.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.
    """
    registry = {}
    dispatch_cache = weakref.WeakKeyDictionary()
    cache_token = None
    registration_class: EnumMeta | None = None

    def dispatch(enum_member):
        """generic_func.dispatch(enum) -> <function implementation>

        Runs the dispatch algorithm to return the implementation
        for the given *enum* registered on *generic_func*. If a special enum member function can't be
        found, it will compare the registered enum type with the initial registration and run the generic function
        instead. Otherwise, raises ValueError.

        """
        nonlocal cache_token
        # enum could be given as string, lookup via conversion, raise Error otherwise
        if isinstance(enum_member, str) and registration_class is not None:
            try:
                enum_member = registration_class[enum_member.upper()]
            except KeyError:
                pass
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token
        try:
            impl = dispatch_cache[enum_member]
        except (KeyError, TypeError):
            try:
                impl = registry[enum_member]
            except KeyError:
                # search for the next higher generic Enum class
                if isinstance(enum_member, registration_class):
                    try:
                        impl = registry[object]
                    except KeyError:
                        raise ValueError(f"I don't know how I got here")
                else:
                    raise ValueError(f"No implementation for {enum_member} registered")
        return impl

    def _is_valid_dispatch_type(enum_member):
        """
        Checks in a generic way if a class or member is of type *Enum*
        """
        return isinstance(enum_member, Enum)

    def _is_valid_dispatch_subtype(enum_member, enum_cls):
        """
        Check if a given enum member is an instance of the pre-registered Enum class and not a class itself
        """
        return isinstance(enum_member, enum_cls) and not isinstance(enum_member, type)

    def register(enum_member, func=None):
        """generic_func.register(enum_member, func) -> func

        Registers a new implementation for the given *enum_member* on a *generic_func*.

        """
        nonlocal cache_token
        if _is_valid_dispatch_type(enum_member):
            # if enum is given in decorator style without typehints e.g. @register(some_enum),
            # return a new decorator with only functional input
            if func is None:
                return lambda f: register(enum_member, f)
        else:
            if func is not None:
                raise TypeError(
                    f"Invalid first argument to `register()`. "
                    f"{enum_member!r} is not an enum member."
                )
            # enum member is now referring to a callable
            ann = getattr(enum_member, '__annotations__', {})
            if not ann:
                raise TypeError(
                    f"Invalid first argument to `register()`: {enum_member!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
            func = enum_member
            # parse function annotations
            # TODO: get_type_hints fails for enums that inherit from 'str' and 'Enum', maybe use __annotation__ directly
            argname, enum_type_hint = next(iter(get_type_hints(func).items()))
            if not _is_valid_dispatch_subtype(enum_type_hint, enum_type_hint.__class__):
                raise TypeError(
                    f"Invalid annotation for {argname!r}. "
                    f"{enum_type_hint!r} is not a member of {enum_type_hint.__class__!r}."
                )
            enum_member = enum_type_hint

        registry[enum_member] = func
        # if cache_token is None and hasattr(cls, '__abstractmethods__'):
        # cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        if not args:
            raise TypeError(f'{funcname} requires at least '
                            '1 positional argument')

        return dispatch(args[0])(*args, **kw)

    funcname = getattr(func, '__name__', 'singledispatch function')
    registry[object] = func
    ann = getattr(func, '__annotations__', {})
    if ann:
        registration_class = next(iter(ann.values()))
    wrapper.registration_class = registration_class
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper


class enumdispatchmethod:
    """
    Generic method descriptor of the `enumdispatch` method.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func!r} is not callable or a descriptor")

        self.dispatcher = enumdispatch(func)
        self.func = func

    def register(self, cls, method=None):
        """generic_method.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            method = self.dispatcher.dispatch(args[0])
            return method.__get__(obj, cls)(*args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, '__isabstractmethod__', False)
