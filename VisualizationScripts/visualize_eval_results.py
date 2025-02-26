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

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.fill(angles, values, color="skyblue", alpha=0.4, edgecolor="blue", linewidth=2)
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_yticklabels([str(y) for y in np.arange(0, 11, 2)])
    ax.set_title("Radar Chart of Average Scores", pad=20)
    plt.show()


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

    outliers = {
        category: [
            score
            for score in scores[category]
            if score
            < np.percentile(scores[category], 25)
            - 3.0
            * (
                np.percentile(scores[category], 75)
                - np.percentile(scores[category], 25)
            )
            or score
            > np.percentile(scores[category], 75)
            + 3.0
            * (
                np.percentile(scores[category], 75)
                - np.percentile(scores[category], 25)
            )
        ]
        for category in categories
    }

    outliers_exist = any(len(v) > 0 for v in outliers.values())

    if outliers_exist:
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
