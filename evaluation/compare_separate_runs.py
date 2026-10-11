import argparse
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare models after training them separately."
    )
    parser.add_argument("--input", type=str, default="artifacts/separate_model_runs.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser


def to_markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )

    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    input_path = project_root / args.input
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Separate run file not found: {input_path}. Train models with --save-run-metrics first."
        )

    df = pd.read_csv(input_path)
    required_cols = {"source", "model", "test_accuracy", "test_f1", "test_roc_auc"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    latest = (
        df.groupby(["source", "model"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    ranked = latest.sort_values(
        by=["test_accuracy", "test_f1", "test_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    csv_path = output_dir / "separate_model_comparison.csv"
    md_path = output_dir / "separate_model_comparison.md"
    ranked.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown_table(ranked), encoding="utf-8")

    print("\n=== Separate Model Comparison ===")
    print(ranked.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison Markdown: {md_path}")


if __name__ == "__main__":
    main()
