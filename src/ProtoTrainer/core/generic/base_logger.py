from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike
from typing import Sequence, BinaryIO, TextIO, TypeVar, get_type_hints, Type

IO_ = TypeVar('IO_', TextIO, BinaryIO)


class Serializer(ABC):
    """
    Basic serializer class used in conjunction with any subtype of `BaseLoggerFactory`.
    Subclasses of `Serializer` are expected to overwrite ``supported_file_ending`` and ``io_type`` as well as the
    abstract functions ``read()`` and ``write()``

    Example::

        class JsonSerializer(Serializer):
            supported_file_ending = '.json'
            io_type = 't'

            @classmethod
            def write(cls, obj: object, filehandler: TextIO, **kwargs):
                json.dump(obj, filehandler, **kwargs)

            @classmethod
            def read(cls, filehandler: TextIO, **kwargs) -> object:
                return json.load(filehandler, **kwargs)

    Attributes:
        supported_file_ending (str): The file ending of the serialized file in dot notation e.g. ``'.json'``
        io_type (str): The filehandler stream type. Either ``'t'`` for text-based streams or ``'b'`` for binary streams
    """

    supported_file_ending: str | Sequence[str]
    io_type: str | None = None

    @classmethod
    @abstractmethod
    def write(cls, obj: object, filehandler: IO_, **kwargs):
        ...

    @classmethod
    @abstractmethod
    def read(cls, filehandler: IO_, **kwargs) -> object:
        ...

    @classmethod
    def deduct_io_type(cls) -> BinaryIO | TextIO | None:
        if cls.io_type is None:
            w_hint = get_type_hints(cls.write).get('filehandler', None)
            r_hint = get_type_hints(cls.read).get('filehandler', None)
            # write gets precedence, None and IO_ are the same case
            if w_hint is not None and w_hint is not IO_:
                return w_hint
            elif r_hint is not None and r_hint is not IO_:
                return r_hint
        return cls.io_type


class BaseLogger:
    """
    The base class for all loggers. Manages paths, file handlers and delegates read/write calls to
    the serializer.

    Attributes:
        serializer (Serializer): The serializer used to handle read/write calls. Must be a subtype of Serializer.
        file_path (PathLike): The path to read and write the serialized file.
        file_exists (bool): Whether the file already exists.
        append_if_exists (bool): Whether to append to already existing files.
        io_type (str): Either ``'t'`` for text-based streams or ``'b'`` for binary streams.
         Depends on the Serializer.
        file_handler (TextIO | BinaryIO): the continuous file handler for write operations. Read calls use a temporary
         file handler instead

    Examples::

        # generate a Json based logger class
        JsonLogger = create_logger(BaseLogger, JsonSerializer)
        # return a logger instance of the JsonLogger
        json_logger = JsonLogger(file_name, write_dir, append_if_exists=True)

        # define a different base class used for file handling
        class CustomLogger(BaseLogger):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def return_handler_mode():
                # always append
                return "a+"

        JsonLogger = create_logger(CustomLogger, JsonSerializer)
        json_logger = JsonLogger('file1', '/tmp', True)
    """

    def __init__(self,
                 file_name: str,
                 serializer: Type[Serializer],
                 write_dir: str | Path | PathLike | None = None,
                 append_if_exists: bool = True):
        """
        Args:
            file_name: name of the file to be written
            serializer: serializer class defining the read/write operation
            write_dir: directory to write the serialized file to
            append_if_exists: whether to append contents if file already exists
        """
        self.serializer = serializer
        if write_dir is None:
            write_dir = Path.cwd()
        else:
            write_dir = Path(write_dir)
        if not write_dir.is_absolute():
            # resolve from cwd
            write_dir = write_dir.resolve()
        Path.mkdir(write_dir, exist_ok=True, parents=True)
        file_path = (write_dir / file_name).with_suffix(serializer.supported_file_ending)
        self.file_path = file_path
        self.file_exists = file_path.exists()
        self.append_if_exists = append_if_exists
        # deduct the io type
        io_type = serializer.io_type if serializer.io_type in {'b', 't'} else None
        if io_type is None:
            # deduct from typehints
            io_type = serializer.deduct_io_type()
            if io_type is None or all((io_type is not BinaryIO, io_type is not TextIO)):
                raise ValueError(f"Can't deduce io type of serializer. Either define serializer.io_type as 'b' or "
                                 f"'t'"
                                 f"or use {TextIO, BinaryIO} type hints in Serializer.read and .write")
            io_type = 'b' if io_type is BinaryIO else 't'
        file_handler_mode = self.return_handler_mode() + io_type
        self.io_type = io_type
        self.file_handler = open(file_path, file_handler_mode)

    def return_handler_mode(self):
        if self.file_exists:
            if self.append_if_exists:
                return 'a'
            else:
                return 'x'  # don't overwrite file silently, throw error
        else:
            return 'w'

    def write(self, obj: object, **kwargs):
        self.serializer.write(obj, self.file_handler, **kwargs)

    def read(self, **kwargs) -> object:
        # open a temporary file handler
        with open(self.file_path, 'r' + self.io_type) as f:
            return self.serializer.read(f, **kwargs)

    def __del__(self):
        self.file_handler.close()


def create_logger(logger_cls: type[BaseLogger], serializer_cls: type[Serializer]):
    class Logger(logger_cls):
        __doc__ = logger_cls.__doc__

        def __init__(
                self,
                file_name: str,
                write_dir: str | Path | PathLike | None = None,
                append_if_exists: bool = True):
            """
            test
            Args:
                file_name: name of the file to be written
                write_dir: directory to write the serialized file to
                append_if_exists: whether to append contents if file already exists
            """

            super().__init__(file_name=file_name,
                             serializer=serializer_cls,
                             write_dir=write_dir,
                             append_if_exists=append_if_exists)

    return Logger
