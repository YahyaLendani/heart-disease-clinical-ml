"""Root compatibility runner for the AI Thesis Reading List experimental suite."""

from experiments.run_reading_list_suite import run_all_advancements

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run complete AI Thesis reading list experimental benchmark suite.")
    parser.add_argument("--heart-csv", type=str, default="cleaned_heart_disease.csv")
    parser.add_argument("--mimic-csv", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()
    run_all_advancements(args.heart_csv, args.mimic_csv, args.output_dir)
