def get_digits(num: int, length: int = None) -> list[int]:
    """
    Get digits of a number

    Args:
        num (int): Number to convert.
        length (int, optional): Length to pad to.

    Returns:
        out (list[int]): A list of digits.
    """
    digits = [int(n) for n in str(num)]

    if length is None:
        padding = []
    else:
        padding = [0] * max(0, length - len(digits))

    return padding + digits


def get_number(digits: list[int]) -> int:
    """
    Convert a list of digits into a single integer.

    Args:
        digits (list[int]): List of digits to convert. Must be non-empty.

    Returns:
        out (int): The integer represented by the digits.
    """
    assert len(digits) > 0, "Cannot convert empty list of digits to a number!"
    return int("".join(map(str, digits)))


def num_to_str(num: int, length: int | None = None) -> str:
    """
    Convert an integer into a space-separated string of its digits.

    Args:
        num (int): The number to convert.
        length (int, optional): Length to pad to.

    Returns:
        out (str): The number as a space-separated string, e.g., 123 -> '1 2 3'.
    """
    digits = get_digits(num)
    if length is None:
        padding = []
    else:
        padding = [0] * max(0, length - len(digits))

    return " ".join(map(str, padding + digits))


def digits_to_str(digits: list[int], length: int | None = None) -> str:
    """
    Convert a list of digits into a space-separated string.

    Args:
        digits (list[int]): List of digits to convert.
        length (int, optional): Length to pad to.

    Returns:
        out (str): Digits as a space-separated string. Returns an empty string if the list is empty.
    """
    if length == 0:
        return ""

    if length is None:
        padding = []
    else:
        padding = [0] * max(0, length - len(digits))

    return " ".join(map(str, padding + digits))
