from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Tuple


class SqlType(Enum):
    INT = auto()
    BIGINT = auto()
    SMALLINT = auto()
    TINYINT = auto()
    BIT = auto()
    FLOAT = auto()
    DATE = auto()
    DATETIME = auto()
    DATETIME2 = auto()
    TIME = auto()
    VARCHAR = auto()
    CHAR = auto()
    DECIMAL = auto()


@dataclass(frozen=True)
class ColumnSpec:
    sql_type: SqlType
    nullable: bool = True
    args: Tuple[Any, ...] = ()


class Dialect(ABC):
    name: ClassVar[str]

    @abstractmethod
    def render_type(self, spec: ColumnSpec) -> str:
        """Return the full column type fragment (including nullability)."""
