import argparse
from typing import List, Optional, Union, Tuple


def int_or_int_list_or_none(value: Optional[Union[int, str]]) -> List[Optional[int]]:
    """
    Parse an input value into a list of integers or a single integer, or None.

    Args:
        value (Optional[Union[int, str]]): The input value to parse.

    Returns:
        List[Optional[int]]: A list containing either a single integer, a list of integers,
                             or a single None value.

    Raises:
        argparse.ArgumentTypeError: If the input value cannot be parsed into the specified formats.
    """
    if value in ['None', 'null']:
        return [None]
    try:
        # If the value contains commas, parse it as a comma-separated list of integers
        if ',' in value:
            return [int(x) for x in value.split(',')]
        # If it's a single integer, pack it into a list
        else:
            return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Use an integer, a comma-separated list of integers, or None.")


def int_or_float(value):
    if '.' in value:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Quality level must be an integer or a float")
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Quality level must be an integer or a float")
