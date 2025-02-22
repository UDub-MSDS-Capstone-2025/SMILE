import argparse
from evaluation import evaluate_dataset
from Utils.json_utils import load_json, save_json


def main():
    """
    Main function to evaluate a dataset using the Gemini API.

    This function parses command-line arguments to get the dataset path and output path,
    loads the dataset, evaluates it using the Gemini API, and saves the evaluation results
    to the specified output file.

    Command-line arguments:
    --dataset_path: Path to the JSON dataset file.
    --output_path: Path to the output file where evaluated scores will be saved (default: "evaluation_results.json").
    """
    parser = argparse.ArgumentParser(description="Evaluate dataset using Gemini API")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the JSON dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluated scores",
    )

    args = parser.parse_args()

    # Load dataset
    conversations = load_json(args.dataset_path)

    # Evaluate dataset
    evaluation_results = evaluate_dataset(conversations)

    # Save results
    save_json(evaluation_results, args.output_path)
    print(f"Evaluation completed. Results saved in {args.output_path}")


if __name__ == "__main__":
    main()
