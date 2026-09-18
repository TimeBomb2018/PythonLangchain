from langchain_core.tools import tool


@tool("calculate", parse_docstring=True)
def calculate(
        a: float,
        b: float,
        operation: str
) -> float:
    """计算两个数的和、差、积、商

    Args:
        a: 数字1
        b: 数字2
        operation: 运算符，支持 add, sub, mul, div

    Returns:
        float: 两个输入数字的运算结果
    """
    if operation == "add":
        return a + b
    elif operation == "sub":
        return a - b
    elif operation == "mul":
        return a * b
    elif operation == "div":
        if b != 0:
            return a / b
        else:
            raise ValueError("除数不能为零")
    else:
        raise ValueError("不支持的运算符")

if __name__ == '__main__':
    result = calculate.invoke({"a": 1, "b": 2, "operation": "div"})
    print(result)
