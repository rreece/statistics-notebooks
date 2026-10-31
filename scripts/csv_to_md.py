#!/usr/bin/env python3
"""
csv_to_md.py
"""


import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infiles",
        default=None,
        nargs="+",
        help="Input csv files.",
    )
    parser.add_argument(
        "--floatfmt",
        default=".2g",
        type=str,
        help="floatfmt used with df.to_markdown (uses .2g for scientific notation on large values)",
    )
    parser.add_argument(
        "--right-align",
        action="store_true",
        help="Right-align data columns (index stays left-aligned).",
    )
    parser.add_argument(
        "--drop-columns",
        default=None,
        type=str,
        help="Comma-separated list of column names to drop before rendering (e.g. for hiding "
             "absolute dollar columns while keeping them in the underlying CSV).",
    )
    return parser.parse_args()


def escape_underscores(text):
    """Escape underscores for LaTeX compatibility, but not inside math mode."""
    parts = re.split(r'(\$[^$]+\$)', text)
    result = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result.append(part)  # Keep math mode as-is
        else:
            result.append(part.replace("_", r"\_"))
    return ''.join(result)


def normalize_separator_dashes(table_str):
    """Normalize markdown table separator dashes to minimal width.

    Pandoc uses the dash lengths in the separator row to calculate column
    proportions in LaTeX output. When headers contain long LaTeX markup
    (e.g. $\\beta_{\\mathrm{MMF}}$), the padding inflates column widths.
    Normalizing to minimal dashes forces Pandoc to auto-size columns.
    """
    lines = table_str.split("\n")
    new_lines = []
    for line in lines:
        # Match separator rows: |:---|---:|:---:|---|
        # These contain only |, -, :, and spaces
        stripped = line.strip()
        if stripped and re.match(r"^[|\s:-]+$", stripped) and "-" in stripped:
            cells = line.split("|")
            new_cells = []
            for cell in cells:
                cell_stripped = cell.strip()
                if not cell_stripped:
                    new_cells.append(cell)
                    continue
                # Preserve alignment markers, minimize dashes
                if cell_stripped.startswith(":") and cell_stripped.endswith(":"):
                    new_cells.append(":---:")
                elif cell_stripped.startswith(":"):
                    new_cells.append(":---")
                elif cell_stripped.endswith(":"):
                    new_cells.append("---:")
                else:
                    new_cells.append("---")
            new_lines.append("|".join(new_cells))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def main():
    args = parse_args()
    infiles = args.infiles
    floatfmt = args.floatfmt

    drop_columns = args.drop_columns.split(",") if args.drop_columns else []

    for fn in infiles:
        bn, ext = os.path.splitext(fn)
        assert ext == ".csv"
        df = pd.read_csv(fn, index_col=0)
        if drop_columns:
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        kwargs = dict(index=True, floatfmt=floatfmt)
        if args.right_align:
            kwargs["colalign"] = ("left",) + ("right",) * len(df.columns)
        s_table = df.to_markdown(**kwargs)
        s_table = escape_underscores(s_table)
        s_table = normalize_separator_dashes(s_table)
        fn_out = bn + ".txt"
        with open(fn_out, "w") as out:
            out.write(s_table)
        print("%s written" % (fn_out))


if __name__ == "__main__":
    main()
