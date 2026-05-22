"""SQL tooling: CREATE TABLE / BULK INSERT generators and the dialect layer.

Sub-modules are imported directly (e.g. ``from src.tools.sql.dialect import ...``);
this package root is intentionally empty so that importing one sub-module does
not eagerly drag in unrelated ones (avoids circular imports between the
dialect package and the generators that consume it).
"""
