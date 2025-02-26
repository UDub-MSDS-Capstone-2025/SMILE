import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from visualize_eval_results import (
    visualize_radar_chart,
    visualize_outliers,
    visualize_score_variability,
    visualize_conversation_scores,
)

from Utils.json_utils import load_json


def main():
    """
    Main function to generate visualizations from evaluation data.
    This function parses command-line arguments to get the JSON file path and the type of visualization to generate.
    It then loads the data from the JSON file and generates the specified visualizations.
    Command-line arguments:
    --json_filepath: Path to the JSON file containing evaluation data.
    --visualization: Type of visualization to generate (default: "all").
    Choices: "all", "radar", "outliers", "variability", "conversation".
    """
    parser = argparse.ArgumentParser(
        description="Generate evaluation score visualizations from a JSON file."
    )
    parser.add_argument(
        "--json_filepath",
        type=str,
        help="Path to the JSON file containing evaluation data.",
    )
    parser.add_argument(
        "--visualization",
        type=str,
        choices=["all", "radar", "outliers", "variability", "conversation"],
        default="all",
        help="Type of visualization to generate.",
    )
    args = parser.parse_args()

    data = load_json(args.json_filepath)

    if args.visualization == "all":
        visualize_radar_chart(data)
        visualize_outliers(data)
        visualize_score_variability(data)
        visualize_conversation_scores(data)
    elif args.visualization == "radar":
        visualize_radar_chart(data)
    elif args.visualization == "outliers":
        visualize_outliers(data)
    elif args.visualization == "variability":
        visualize_score_variability(data)
    elif args.visualization == "conversation":
        visualize_conversation_scores(data)


if __name__ == "__main__":
    main()
