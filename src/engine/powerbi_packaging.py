# Assumes PBIP templates define a Power Query expression named:
#   expression ContosoFolder
# used as the root path for generated datasets.

import shutil
import re
from pathlib import Path


def attach_pbip_project(
    final_folder: Path,
    pbip_template_root: Path,
    expression_name: str = "ContosoFolder",
):
    """
    Copies a PBIP template into the final output folder and
    rewrites the specified TMDL expression to point at final_folder.
    """

    # ------------------------------------------------------------
    # Copy PBIP folder into final output
    # ------------------------------------------------------------
    pbip_dst = final_folder / pbip_template_root.name

    if pbip_dst.exists():
        shutil.rmtree(pbip_dst)

    shutil.copytree(pbip_template_root, pbip_dst)

    # ------------------------------------------------------------
    # Locate SemanticModel folder (auto-detect)
    # ------------------------------------------------------------
    semantic_models = list(pbip_dst.glob("*.SemanticModel"))

    if len(semantic_models) != 1:
        raise RuntimeError(
            f"Expected exactly one *.SemanticModel folder in PBIP, "
            f"found {len(semantic_models)}"
        )

    semantic_model = semantic_models[0]

    expressions_file = (
        semantic_model
        / "definition"
        / "expressions.tmdl"
    )

    if not expressions_file.exists():
        raise RuntimeError(
            "expressions.tmdl not found. Ensure the PBIP template defines "
            f"an expression named '{expression_name}'.\n"
            f"Expected path: {expressions_file}"
        )

    # ------------------------------------------------------------
    # Rewrite the expression
    # ------------------------------------------------------------
    _rewrite_expression(
        expressions_file=expressions_file,
        expression_name=expression_name,
        final_folder=final_folder,
    )


def _rewrite_expression(
    *,
    expressions_file: Path,
    expression_name: str,
    final_folder: Path,
):
    """
    Rewrite exactly one expression of the form:

        expression <expression_name> = "<path>"

    Replacing <path> with final_folder.
    """

    text = expressions_file.read_text(encoding="utf-8")
    new_path = str(final_folder)

    pattern = rf'''
        expression\s+{re.escape(expression_name)}\s*=\s*
        ".*?"
    '''

    updated, count = re.subn(
        pattern,
        lambda _: f'expression {expression_name} = "{new_path}"',
        text,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one expression named '{expression_name}', "
            f"found {count}"
        )

    expressions_file.write_text(updated, encoding="utf-8")
