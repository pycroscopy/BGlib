"""
MCP server exposing a small, focused BGlib toolset.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import logging
from pathlib import Path
from typing import Any

TOOL_NAMES = [
    "LabViewPatcher",
    "projectLoop",
    "calc_switching_coef_vec",
    "calculate_loop_centroid",
    "get_rotation_matrix",
]

logger = logging.getLogger(__name__)


def _import_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from fastmcp import FastMCP
        except ImportError as exc:
            raise RuntimeError(
                "MCP support is not installed. Install BGlib with the optional "
                "'mcp' extra, for example: pip install -e .[mcp]"
            ) from exc
    return FastMCP


def _to_jsonable(value: Any) -> Any:
    np = None
    try:
        import numpy as np  # type: ignore
    except ImportError:
        pass

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.void) and value.dtype.names:
            return {name: _to_jsonable(value[name]) for name in value.dtype.names}
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    return value


@lru_cache(maxsize=1)
def _load_be_loop_module():
    module_path = (
        Path(__file__).resolve().parent / "BGlib" / "be" / "analysis" / "utils" / "be_loop.py"
    )
    spec = spec_from_file_location("_bglib_be_loop_mcp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load BGlib be_loop module for MCP server.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_labview_patcher_class():
    module = import_module("BGlib.be.translators.labview_h5_patcher")
    return module.LabViewH5Patcher


def _build_server():
    FastMCP = _import_fastmcp()
    mcp = FastMCP("BGlib")

    @mcp.tool()
    def LabViewPatcher(h5_path: str, force_patch: bool = False) -> dict[str, Any]:
        """
        Patch a LabView-generated HDF5 file in place so it is Pycroscopy-ready.
        """
        patcher_cls = _load_labview_patcher_class()
        patched_path = patcher_cls().translate(h5_path, force_patch=force_patch)
        return {"patched_path": patched_path, "force_patch": force_patch}

    @mcp.tool()
    def projectLoop(
        vdc: list[float],
        amp_vec: list[float],
        phase_vec: list[float],
    ) -> dict[str, Any]:
        """
        Project a single loop cycle from DC voltage, amplitude, and phase vectors.
        """
        be_loop = _load_be_loop_module()
        results = be_loop.projectLoop(vdc, amp_vec, phase_vec)
        return _to_jsonable(results)

    @mcp.tool()
    def calc_switching_coef_vec(
        loop_coef_vec: list[list[float]],
        nuc_threshold: float,
    ) -> dict[str, Any]:
        """
        Convert loop-fit coefficients into physical switching parameters.
        """
        be_loop = _load_be_loop_module()

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to use calc_switching_coef_vec.") from exc

        result = be_loop.calc_switching_coef_vec(
            np.asarray(loop_coef_vec, dtype=float),
            nuc_threshold,
        )
        return {"switching_coefficients": [_to_jsonable(row) for row in result]}

    @mcp.tool()
    def calculate_loop_centroid(
        vdc: list[float],
        loop_vals: list[float],
    ) -> dict[str, Any]:
        """
        Calculate the polygon centroid and geometric area for one loop.
        """
        be_loop = _load_be_loop_module()
        centroid, area = be_loop.calculate_loop_centroid(vdc, loop_vals)
        return {"centroid": _to_jsonable(centroid), "area": _to_jsonable(area)}

    @mcp.tool()
    def get_rotation_matrix(theta: float) -> dict[str, Any]:
        """
        Return the 2D rotation matrix for an angle in radians.
        """
        be_loop = _load_be_loop_module()
        matrix = be_loop.get_rotation_matrix(theta)
        return {"rotation_matrix": _to_jsonable(matrix)}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the BGlib MCP server over stdio. "
            "This command is intended to be launched by an MCP client."
        )
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the names of the exposed MCP tools and exit.",
    )
    args = parser.parse_args()

    if args.list_tools:
        print("\n".join(TOOL_NAMES))
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("BGlib MCP server starting on stdio; waiting for an MCP client...")
    server = _build_server()
    server.run()


if __name__ == "__main__":
    main()
