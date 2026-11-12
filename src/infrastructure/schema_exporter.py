from functools import lru_cache
from typing import Any, Dict
from loguru import logger
from sqlalchemy import Enum
from sqlalchemy.dialects import postgresql
from src.infrastructure.models import Base


def _resolve_column_default(col: Any) -> Any:
    if col.server_default is not None and hasattr(col.server_default, "arg"):
        return str(col.server_default.arg)
    if col.default is not None and hasattr(col.default, "arg"):
        if callable(col.default.arg):
            return "CURRENT_TIMESTAMP"
        return col.default.arg
    return None


def _resolve_column_type(col: Any) -> str:
    if isinstance(col.type, Enum):
        return "ENUM"
    return col.type.compile(dialect=postgresql.dialect())


@lru_cache(maxsize=1)
def export_schema_dict() -> Dict[str, Any]:
    logger.debug("Extracting structured database schema from SQLAlchemy metadata.")
    tables_data: Dict[str, Any] = {}
    for table in Base.metadata.sorted_tables:
        cols = [
            {
                "name": col.name,
                "type": _resolve_column_type(col),
                "nullable": bool(col.nullable),
                "primary_key": bool(col.primary_key),
                "default": _resolve_column_default(col),
                "enum_values": list(col.type.enums) if isinstance(col.type, Enum) else None,
            }
            for col in table.columns
        ]
        fks = [
            {
                "column": fk.parent.name,
                "target": fk.target_fullname,
                "ondelete": fk.ondelete,
            }
            for fk in table.foreign_keys
        ]
        idxs = [
            {
                "name": idx.name,
                "columns": [c.name for c in idx.columns],
                "unique": bool(idx.unique),
            }
            for idx in table.indexes
        ]
        tables_data[table.name] = {
            "columns": cols,
            "foreign_keys": fks,
            "indexes": idxs,
        }
    logger.info(f"Structured schema extracted successfully ({len(tables_data)} tables).")
    return tables_data
