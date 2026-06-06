from langchain_core.tools import tool


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform one arithmetic operation on two numbers. Must be used for all math questions.

    Supported operations: add, sub, mul, div.
    For multi-step expressions, call this tool once per operation (e.g. 5*10, then +5, then -5, etc.).
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}
