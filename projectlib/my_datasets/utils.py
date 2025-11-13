
def get_digits(num: int) -> list[int]:
    """
    Get digits of a number

    Args:
        num (int): Number to convert.

    Returns:
        out (list[int]): A list of digits.
    """
    return [int(n) for n in str(num)]


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


def num_to_str(num: int) -> str:
    """
    Convert an integer into a space-separated string of its digits.

    Args:
        num (int): The number to convert.

    Returns:
        out (str): The number as a space-separated string, e.g., 123 -> '1 2 3'.
    """
    return " ".join(map(str, get_digits(num)))


def digits_to_str(digits: list[int]) -> str:
    """
    Convert a list of digits into a space-separated string.

    Args:
        digits (list[int]): List of digits to convert.

    Returns:
        out (str): Digits as a space-separated string. Returns an empty string if the list is empty.
    """
    if len(digits) == 0:
        return ""
    return " ".join(map(str, digits))


def bb_tokinpt_to_str(board: str, pad_token: str, row_sep_token: str) -> str:
    """Convert a blackboard state encoded in a tokenizer-friendly format into a well-readable string.

    Args:
        board (str): The blackboard state encoded in a tokenizer-friendly format
        pad_token (str): The token used to pad the board
        row_sep_token (str): The token used to separate rows

    Returns:
        out (str): The blackboard state as a printable string.
    """

    out = ""
    rows = board.replace(' ', '').replace(pad_token, ' ').split(row_sep_token)[:-1]
    out += "-" * (len(rows[0]) + 2) + "\n"
    for row in rows:
        out += f"|{row}|\n"
    out += "-" * (len(rows[0]) + 2) + "\n"

    return out
