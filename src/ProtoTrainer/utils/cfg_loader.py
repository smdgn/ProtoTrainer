import yaml
from collections import abc
from pathlib import Path
import argparse
from ProtoTrainer.core.generic.base_enum import BaseEnum


def parse_yaml_to_config(yaml_file: Path | list[Path],
                         attribute_access: bool = False) -> dict:
    """
    Takes a list of yaml files and parses them into a named tuple for attribute style access.
    If a single directory is provided, it will search in ascending order for yaml files and parse all,
    otherwise it will just parse one yaml file.
    :param yaml_file: A singular path to a folder or a list of yaml files."""
    if attribute_access:
        # cfg_subobj.append(namedtuple(file.stem, yaml_dict.keys())(*yaml_dict.values()))
        raise NotImplementedError("Not implemented for attribute access")

    if isinstance(yaml_file, list):
        files = yaml_file
    else:
        if yaml_file.is_file():
            files = [yaml_file]
        else:
            # is a directory
            files = list(yaml_file.glob('./*.yaml'))

    cfg_subobj = {}
    for file in files:
        with open(file, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        # add to dict
        cfg_subobj[file.stem] = yaml_dict

    return cfg_subobj


class ConfigParser:
    """
    Combines argparse with yaml configs

    Attributes:
        config_names: List of yaml file names that are included in the config dict.
            You can access the values of the added .yaml configs with ``self.{name}_config``
        _parser: the argparse instance

    """

    def __init__(self,
                 name: str,
                 prog: str | None = None,
                 description: str | None = None,
                 epilog: str | None = None,
                 config_path: list[Path] | Path | None = None,
                 config_names: list[str] | str | None = None
                 ):
        """
        Args:
            name (str): the unique identifier for this _parser instance. Necessary if multiple parsers are merged.
            config_path (list): List of yaml file paths. Can be a single path or a list of paths.
                If the single path points to a directory, it will search in ascending order for yaml files and parse all.
                If the single path points to a file, only the yaml file will be loaded.
            config_names (list): List of yaml file names that are included in the config dict.
                You can access the values of the added .yaml configs with ``self.{name}_config``


        """
        self._parser = argparse.ArgumentParser(description=description, epilog=epilog, prog=prog)
        self._actions = []
        self._subparsers = None
        self.name = name

        self.config_names = [config_names] if isinstance(config_names, str) else config_names
        if config_path is not None:
            self.cfg = parse_yaml_to_config(config_path)

            if self.config_names is not None:
                self._check_cfg()
                names = self.config_names
            else:
                names = self.cfg.keys()
            for name in names:
                self.__setattr__(name + "_config", self.cfg[name])

    @property
    def configs(self):
        return {key: value for key, value in vars(self).items() if key.endswith("_config")}

    def _check_cfg(self):
        # check if the config names are given in the config path
        values_valid = all([name in self.cfg for name in self.config_names])
        if not values_valid:
            raise ValueError("Number of needed config names not received. Is the naming correct?")

    def create_choices_for(self, arg_names: list[str], enum_types: list[BaseEnum],
                           help_description: list[str] | None = None):
        assert len(arg_names) == len(enum_types), f"Length of config names and enum types do not match"
        if help_description is not None:
            assert len(help_description) == len(arg_names), f"Length of help list and config names do not match"
        else:
            help_description = [config_name for config_name in arg_names]
        for config_name, enum_type, desc in zip(arg_names, enum_types, help_description):
            self.add_argument(f"{config_name}", type=str, choices=BaseEnum.choices, help=desc)
        return self

    def add_yaml_flags(self, cfg: dict):
        def nested_dict_iter(nested_dict, flag_trajectory):
            for key, value in nested_dict.items():
                current_path = flag_trajectory + [key]
                if isinstance(value, abc.Mapping):
                    nested_dict_iter(value, current_path)
                else:
                    flag_paths.append(current_path)

        flag_paths = []
        nested_dict_iter(cfg, [])
        return flag_paths

    def add_argument(
            self,
            *args,
            **kwargs,
    ):
        # TODO: maybe adjust parameter names
        self._actions.append({
            "args": args,
            "kwargs": kwargs})
        args = self._actions[-1]
        self._parser.add_argument(*args["args"], **args["kwargs"])

    def parse_args(self, args=None, namespace=None):
        self._parser.parse_args(args, namespace)

    def __add__(self, other: "ConfigParser"):

        # get the attributes of other
        other_config = other.configs
        if any(key in self.configs.keys() for key in other_config.keys()):
            raise ValueError("Clashing config names")

        for other_key, other_value in other_config.items():
            self.__setattr__(other_key, other_value)

        if not self._subparsers:
            self._subparsers = self._parser.add_subparsers(dest="subparser")

        parser_b = self._subparsers.add_parser(other.name)

        # merge parsers, other will be the subparser of self
        if other._actions:
            _ = [parser_b.add_argument(*arguments["args"], **arguments["kwargs"]) for arguments in other._actions]

        return self
