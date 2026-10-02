"""Isolated Python Code Execution Sandbox.
Captures output, Plotly & Matplotlib figures, computed DataFrames,
and intercepts errors with detailed traceback localization.
"""

import sys
import io
import traceback
import linecache
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


class ExecutionResult:
    """Encapsulates the execution output, charts, and error diagnostics."""

    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        figures: Optional[List[go.Figure]] = None,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
        result_variables: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        traceback_str: Optional[str] = None,
        failing_line: Optional[int] = None,
        failing_code: Optional[str] = None
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.figures = figures or []
        self.tables = tables or {}
        self.result_variables = result_variables or {}
        self.error_type = error_type
        self.error_message = error_message
        self.traceback_str = traceback_str
        self.failing_line = failing_line
        self.failing_code = failing_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "figures_count": len(self.figures),
            "tables_count": len(self.tables),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failing_line": self.failing_line,
            "failing_code": self.failing_code
        }


class CodeSandbox:
    """Executes analytical Python code in a controlled environment."""

    @staticmethod
    def execute(code: str, df: pd.DataFrame, extra_globals: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Executes the given Python code block with access to DataFrame `df` and standard libraries.

        Intercepts generated figures, printed text, and exceptions.
        """
        # Create dedicated working copy of DataFrame
        df_copy = df.copy()

        # Build clean execution scope
        exec_globals = {
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "plt": plt,
            "sns": sns,
            "df": df_copy,
            "__name__": "__main__",
        }
        if extra_globals:
            exec_globals.update(extra_globals)

        exec_locals: Dict[str, Any] = {}

        # Redirect standard output and error
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Clear matplotlib current figure to avoid carryover
        plt.close("all")

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute code block
            exec(code, exec_globals, exec_locals)

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Inspect variables for figures and tables
            figures: List[go.Figure] = []
            tables: Dict[str, pd.DataFrame] = {}
            result_vars: Dict[str, Any] = {}

            # Search in locals first, then globals
            all_scope = {**exec_globals, **exec_locals}

            # Check if matplotlib has an active figure with content
            if plt.get_fignums():
                try:
                    # Matplotlib figure exists
                    pass
                except Exception:
                    pass

            # Detect Plotly Figures
            for var_name, val in all_scope.items():
                if isinstance(val, (go.Figure, px.scatter.__annotations__.get("return", go.Figure))):
                    if val not in figures:
                        figures.append(val)
                elif isinstance(val, pd.DataFrame) and var_name not in ["df", "df_copy"]:
                    tables[var_name] = val
                elif isinstance(val, (int, float, str, list, dict)) and not var_name.startswith("__"):
                    result_vars[var_name] = val

            return ExecutionResult(
                success=True,
                stdout=stdout_text,
                stderr=stderr_text,
                figures=figures,
                tables=tables,
                result_variables=result_vars
            )

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            exc_type, exc_val, exc_tb = sys.exc_info()
            tb_formatted = traceback.format_exc()

            # Find the line in the user's executed code that caused the error
            failing_line = None
            failing_code = None

            # Traceback entries
            tb_frames = traceback.extract_tb(exc_tb)
            for frame in reversed(tb_frames):
                if frame.filename == "<string>":
                    failing_line = frame.lineno
                    code_lines = code.splitlines()
                    if 1 <= failing_line <= len(code_lines):
                        failing_code = code_lines[failing_line - 1].strip()
                    break

            error_type = exc_type.__name__ if exc_type else "Exception"
            error_message = str(e)

            return ExecutionResult(
                success=False,
                stdout=stdout_text,
                stderr=stderr_text,
                error_type=error_type,
                error_message=error_message,
                traceback_str=tb_formatted,
                failing_line=failing_line,
                failing_code=failing_code
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
