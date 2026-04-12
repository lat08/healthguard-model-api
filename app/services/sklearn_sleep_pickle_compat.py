"""Unpickle compatibility for sleep artifacts built with scikit-learn 1.6.x.

Bundles may reference ``_RemainderColsList`` on ``sklearn.compose._column_transformer``;
that helper was removed in scikit-learn 1.7+. Injecting it before ``joblib.load`` restores
loading without re-exporting the model.

Implementation matches scikit-learn 1.6.1 (BSD-3-Clause).
"""

from __future__ import annotations

import warnings
from collections import UserList


class _RemainderColsList(UserList):
    """List used for ColumnTransformer remainder column metadata (legacy pickles)."""

    def __init__(
        self,
        columns,
        *,
        future_dtype=None,
        warning_was_emitted=False,
        warning_enabled=True,
    ):
        super().__init__(columns)
        self.future_dtype = future_dtype
        self.warning_was_emitted = warning_was_emitted
        self.warning_enabled = warning_enabled

    def __getitem__(self, index):
        self._show_remainder_cols_warning()
        return super().__getitem__(index)

    def _show_remainder_cols_warning(self):
        if self.warning_was_emitted or not self.warning_enabled:
            return
        self.warning_was_emitted = True
        future_dtype_description = {
            "str": "column names (of type str)",
            "bool": "a mask array (of type bool)",
            None: "a different type depending on the ColumnTransformer inputs",
        }.get(self.future_dtype, self.future_dtype)

        warnings.warn(
            (
                "\nThe format of the columns of the 'remainder' transformer in"
                " ColumnTransformer.transformers_ will change in version 1.7 to"
                " match the format of the other transformers.\nAt the moment the"
                " remainder columns are stored as indices (of type int). With the same"
                " ColumnTransformer configuration, in the future they will be stored"
                f" as {future_dtype_description}.\nTo use the new behavior now and"
                " suppress this warning, use"
                " ColumnTransformer(force_int_remainder_cols=False).\n"
            ),
            category=FutureWarning,
            stacklevel=3,
        )

    def _repr_pretty_(self, printer, *_):
        printer.text(repr(self.data))


def patch_sklearn_column_transformer_for_legacy_sleep_pickle() -> None:
    """Ensure ``_RemainderColsList`` exists on sklearn's module before unpickling."""
    import sklearn.compose._column_transformer as ct  # noqa: PLC0415

    if hasattr(ct, "_RemainderColsList"):
        return
    ct._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
