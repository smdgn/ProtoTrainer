import torch
import json

from typing import BinaryIO, TextIO

from src.ProtoTrainer.core.generic.base_logger import Serializer, BaseLogger, create_logger


class _JsonSerializer(Serializer):
    supported_file_ending = '.json'
    io_type = 't'

    @classmethod
    def write(cls, obj: object, filehandler: TextIO, **kwargs):
        json.dump(obj, filehandler, **kwargs)

    @classmethod
    def read(cls, filehandler: TextIO, **kwargs) -> object:
        return json.load(filehandler, **kwargs)


class _TorchSerializer(Serializer):
    supported_file_ending = '.pt'
    io_type = 'b'

    @classmethod
    def write(cls, obj: object, filehandler: BinaryIO, **kwargs):
        torch.save(obj, filehandler)

    @classmethod
    def read(cls, filehandler: BinaryIO, **kwargs) -> object:
        return torch.load(filehandler, **kwargs)


JsonLogger = create_logger(BaseLogger, _JsonSerializer)
TorchLogger = create_logger(BaseLogger, _TorchSerializer)