from ProtoTrainer.utils.dispatcher import enumdispatchmethod
from enum import Enum
import pytest


class MyEnum(Enum):
    ENUM1 = "enum1"
    ENUM2 = "enum2"
    GENERIC = "generic"


class AnotherEnum(Enum):
    ENUM1 = "enum1"


class X:
    def __init__(self):
        pass

    @enumdispatchmethod
    def dispatch(self, type: MyEnum):
        print("I am a generic dispatcher for all MyEnum instances")
        return type, 0

    @dispatch.register
    def _(self, type: MyEnum.ENUM1):
        print("I am a MyEnum.ENUM1 dispatcher")
        return type, 1

    @dispatch.register
    def _(self, type: MyEnum.ENUM2):
        print("I am a MyEnum.ENUM2 dispatcher")
        return type, 2


x = X()


def test_enumdispatchmethod_on_generic():
    type, number = x.dispatch(MyEnum.GENERIC)
    assert isinstance(type, MyEnum)
    assert type == MyEnum.GENERIC
    assert number == 0


def test_enumdispatchmethod_on_enum1():
    type, number = x.dispatch(MyEnum.ENUM1)
    assert isinstance(type, MyEnum)
    assert type == MyEnum.ENUM1
    assert number == 1


def test_enumdispatchmethod_on_enum2():
    type, number = x.dispatch(MyEnum.ENUM2)
    assert isinstance(type, MyEnum)
    assert type == MyEnum.ENUM2
    assert number == 2


def test_enumdispatchmethod_on_enum_str():
    type, number = x.dispatch("enum2")
    assert isinstance(type, str)
    assert number == 2


def test_enumdispatchmethod_on_non_enum_str():
    with pytest.raises(ValueError):
        type, number = x.dispatch("enum3")


def test_enumdispatchmethod_on_anotherenum():
    with pytest.raises(ValueError):
        type, number = x.dispatch(AnotherEnum.ENUM1)


def test_enumdispatchmethod_on_int():
    with pytest.raises(ValueError):
        type, number = x.dispatch(1)
