import enum

__all__ = ["Type"]


class Type(enum.StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    TIME = "time"
    JSON = "json"
    ARRAY = "array"
