import matplotlib.pyplot as plt
import numpy as np


def calculate_average_scores(data):
    """
    Calculates the average scores for each category across all conversations.
    This function iterates through the evaluation scores of each conversation,
    computes the average score for each category, and returns the categories
    along with their corresponding average scores.
    :param data: List of conversation data, where each conversation contains evaluation scores.
    :return: Tuple containing a list of categories and a list of average scores for each category.
    """
    categories = list(data[0]["evaluation_scores"].keys())
    average_scores = [
        np.mean([conv["evaluation_scores"][category]["score"] for conv in data])
        for category in categories
    ]
    # Print average scores for each category
    for category, avg_score in zip(categories, average_scores):
        print(f"Category: {category}, Average Score: {avg_score:.2f}")
    return categories, average_scores


def visualize_radar_chart(data):
    """
    Visualizes the average scores for each category using a radar chart.
    This function calculates the average score for each category across all conversations
    and plots a radar chart to show the distribution of these average scores.
    :param data: List of conversation data, where each conversation contains evaluation scores.
    :return: None
    """
    categories, average_scores = calculate_average_scores(data)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = average_scores + average_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.fill(angles, values, color="skyblue", alpha=0.4, edgecolor="blue", linewidth=2)
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_yticklabels(
        [str(y) for y in np.arange(0, 11, 2)], fontsize=10, fontweight="bold"
    )
    ax.set_title(
        "Radar Chart of Average Scores", pad=20, fontsize=16, fontweight="bold"
    )
    plt.show()


def save_outliers_to_file(outlier_ids, filename="outliers.txt"):
    with open(filename, "w") as file:
        for conv_id in outlier_ids:
            file.write(f"{conv_id}\n")


def visualize_outliers(data):
    """
    Visualizes the number of outliers in each category.
    This function identifies outliers in the evaluation scores for each category
    and plots a bar chart showing the count of outliers per category. An outlier is defined
    as a score that is either below the 25th percentile minus 1.5 times the interquartile range
    or above the 75th percentile plus 1.5 times the interquartile range.
    :param data: List of conversation data, where each conversation contains evaluation scores.
    :return: None
    """
    categories = list(data[0]["evaluation_scores"].keys())
    scores = {
        category: [conv["evaluation_scores"][category]["score"] for conv in data]
        for category in categories
    }

    outlier_ids = []
    outliers = {
        category: [
            (conv["conversation_id"], score)
            for conv, score in zip(data, scores[category])
            if score
            < np.percentile(scores[category], 25)
            - 1.5
            * (
                np.percentile(scores[category], 75)
                - np.percentile(scores[category], 25)
            )
            or score
            > np.percentile(scores[category], 75)
            + 1.5
            * (
                np.percentile(scores[category], 75)
                - np.percentile(scores[category], 25)
            )
        ]
        for category in categories
    }

    for category, values in outliers.items():
        outlier_ids.extend([conv_id for conv_id, _ in values])

    outliers_exist = any(len(v) > 0 for v in outliers.values())

    if outliers_exist:
        save_outliers_to_file(set(outlier_ids))
        plt.figure(figsize=(8, 6))
        plt.bar(outliers.keys(), [len(v) for v in outliers.values()], color="red")
        plt.title("Outlier Counts Per Category")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No outliers detected.")


def visualize_score_variability(data):
    """
    Visualizes the variability of scores across different categories.
    This function calculates the distribution of scores for each category
    and plots a boxplot to show the variability of scores across categories.
    :param data: List of conversation data, where each conversation contains evaluation scores.
    :return: None
    """
    categories = list(data[0]["evaluation_scores"].keys())
    scores = {
        category: [conv["evaluation_scores"][category]["score"] for conv in data]
        for category in categories
    }

    plt.figure(figsize=(8, 6))
    plt.boxplot(scores.values(), labels=scores.keys())
    plt.title("Score Variability Across Categories")
    plt.xticks(rotation=45)
    plt.show()


def visualize_conversation_scores(data):
    """
    Visualizes the distribution of average scores for each conversation.
    This function calculates the average score for each conversation across all categories
    and plots a histogram to show the distribution of these average scores.
    :param data: List of conversation data, where each conversation contains evaluation scores.
    :return: None
    """
    categories = list(data[0]["evaluation_scores"].keys())
    conversation_avg_scores = [
        np.mean([conv["evaluation_scores"][cat]["score"] for cat in categories])
        for conv in data
    ]

    plt.figure(figsize=(8, 6))
    plt.hist(
        conversation_avg_scores, bins=10, color="blue", alpha=0.7, edgecolor="black"
    )
    plt.title("Histogram of Average Conversation Scores")
    plt.xlabel("Average Score")
    plt.ylabel("Number of Conversations")
    plt.show()
