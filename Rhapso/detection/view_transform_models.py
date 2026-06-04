from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

"""
View Transform Models parses and combines view registrations matrices.
"""

class ViewTransformModels:
    def __init__(self, df: dict[str, Any]):
        self.view_registrations_df = df.get(
            "view_registrations", pd.DataFrame()
        )
        self.calibration_matrices: dict[str, dict[str, np.ndarray]] = {}
        self.rotation_matrices: dict[str, dict[str, np.ndarray]] = {}
        self.concatenated_matrices: dict[str, np.ndarray] = {}
    
    @staticmethod
    def _parse_affine_3x4(affine_text: Any, *, view_id: str) -> np.ndarray:
        """
        Parse an affine string containing 12 values into a 4x4 matrix.
        """
        clean_text = str(affine_text).replace(",", " ").strip()

        vals = np.fromstring(
            clean_text,
            sep=" ",
            dtype=np.float64,
        )

        if len(vals) != 12:
            raise ValueError(
                f"{view_id}: expected 12 affine values, got {len(vals)}. "
                f"affine_text={affine_text!r}"
            )

        m = np.eye(4, dtype=np.float64)
        m[:3, :4] = vals.reshape(3, 4)
        return m

    def _view_id(self, timepoint: Any, setup: Any) -> str:
        return f"timepoint: {timepoint}, setup: {setup}"

    def compose_all_view_transforms(self) -> None:
        """
        Compose a per-view 4x4 by chaining all affine transforms in order.
        Running chain per view: M = M @ Ti
        """
        if self.view_registrations_df.empty:
            raise ValueError("view_registrations_df is empty")

        df = self.view_registrations_df[
            self.view_registrations_df["type"] == "affine"
        ].copy()

        sort_cols = ["timepoint", "setup"]
        if "order" in df.columns:
            sort_cols.append("order")
        df = df.sort_values(sort_cols)

        out: dict[str, np.ndarray] = {}

        for (tp, setup), g in df.groupby(
            ["timepoint", "setup"],
            sort=False,
        ):
            view_id = self._view_id(tp, setup)
            m = np.eye(4, dtype=np.float64)

            for _, row in g.iterrows():
                ti = self._parse_affine_3x4(
                    row.get("affine"),
                    view_id=view_id,
                )
                m = m @ ti

            out[view_id] = m

        self.concatenated_matrices = out

    def run(self) -> dict[str, np.ndarray]:
        """
        Execute the entry point of the script.
        """
        self.compose_all_view_transforms()
        return self.concatenated_matrices
