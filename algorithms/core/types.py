from typing import Optional

class Type:
    """Base type."""
    pass

class PrimitiveType(Type):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"PrimitiveType({self.name})"

class ContainerType(Type):
    def __init__(self, element_type: Type, size: Optional[int] = None):
        self.element_type = element_type
        self.size = size  # None = dynamic

    def __repr__(self):
        size_str = f", size={self.size}" if self.size is not None else ""
        return f"ContainerType({self.element_type}{size_str})"

# Standard types
INT = PrimitiveType("int")
FLOAT = PrimitiveType("float")
BOOL = PrimitiveType("bool")
STRING = PrimitiveType("str")

def LIST(element_type: Type) -> ContainerType:
    return ContainerType(element_type)

def DICT(key_type: Type, value_type: Type) -> ContainerType:
    # For simplicity, represent dict as container of pairs
    return ContainerType((key_type, value_type))
