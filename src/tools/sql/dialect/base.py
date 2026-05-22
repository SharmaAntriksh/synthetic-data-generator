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
    # Batch-script terminator (e.g. SQL Server's "GO"). Empty for dialects
    # that have no equivalent — generators skip the line when empty.
    batch_separator: ClassVar[str] = ""

    @abstractmethod
    def render_type(self, spec: ColumnSpec) -> str: ...

    @abstractmethod
    def quote_ident(self, name: str) -> str: ...

    @abstractmethod
    def drop_table_if_exists(self, schema: str, table: str) -> str:
        """Return the dialect-appropriate "drop this table if it exists" statement.

        Takes unquoted identifiers — the dialect handles its own quoting and
        any escaping needed for embedded string literals.
        """
