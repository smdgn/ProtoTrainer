import sys

from src.ProtoTrainer.utils.cfg_loader import ConfigParser

# import pytest

main_parser = ConfigParser("main", "test_configparser.py",
                           "Test script of ConfigParser", "test epilog")
main_parser.add_argument("pos1", type=int, help="help text for positional argument pos1", default=0)
main_parser.add_argument("pos2", type=str, help="help text for positional argument pos2", default="hi")
main_parser.add_argument("-o", "--opt", type=int, help="help text for optional argument opt1", default=0)

aux_parser = ConfigParser("aux")
aux_parser.add_argument("-a" "--aux", type=int, help="help text for optional argument aux", default=0)

mixed_parser = main_parser + aux_parser

if __name__ == "__main__":
    print(sys.argv[1:])
    args = mixed_parser.parse_args(sys.argv[1:])
    print(args)
