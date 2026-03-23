"""Employee dimension package.

Submodules:
  generator                   — employee hierarchy and staffing
  employee_store_assignments  — bridge table (home store assignments)
  transfers                   — optional transfer engine (Phase 3)

Re-exports the public API so existing ``from src.dimensions.employees import ...``
statements continue to work.
"""
from src.dimensions.employees.generator import (           # noqa: F401
    generate_employee_dimension,
    run_employees,
    STORE_MGR_KEY_BASE,
    STAFF_KEY_BASE,
    STAFF_KEY_STORE_MULT,
)
from src.dimensions.employees.employee_store_assignments import (  # noqa: F401
    generate_employee_store_assignments,
    run_employee_store_assignments,
)
