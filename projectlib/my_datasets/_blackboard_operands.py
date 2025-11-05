from abc import ABC, abstractmethod

class CarryOperation(ABC):

    @abstractmethod
    def step(self, op1: int, op2: int, last_carry: int) -> tuple[int, int]:
        """Perform a basic operation on two numbers.

        Args:
            op1 (Number): The first operand.
            op2 (Number): The second operand.
            last_carry (Number): The carry from the last operation.

        Returns:
            tuple[Number, Number]: (res, carry) The result of the operation and the carry.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the operation that should be used in the dataset name."""
        pass

    @abstractmethod
    def finalize_carry(self, carry: int) -> str:
        """Decides what to do with the last carry after the operation is complete.

        Args:
            carry (Number): The carry from the last operation.

        Returns:
            Number: The final result symbol.
        """
        pass

class Addition(CarryOperation):

    def step(self, op1: int, op2: int, last_carry: int) -> tuple[int, int]:
        return (op1 + op2 + last_carry) % 10, (op1 + op2 + last_carry) // 10

    def __str__(self) -> str:
        return "+"

    def get_name(self) -> str:
        return "addition"

    def finalize_carry(self, carry: int) -> str:
        return str(carry)

class Subtraction(CarryOperation):

    def step(self, op1: int, op2: int, last_carry: int) -> tuple[int, int]:
        diff = op1 - op2 - last_carry
        if diff < 0:
            return 10 + diff, 1
        else:
            return diff, 0

    def __str__(self) -> str:
        return "-"

    def get_name(self) -> str:
        return "-"

    def finalize_carry(self, carry: int) -> str:
        if carry:
            raise ValueError("Invalid carry. Please ensure that op1 >= op2")
        return "0"
