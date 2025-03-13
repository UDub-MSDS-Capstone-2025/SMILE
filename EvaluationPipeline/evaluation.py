import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from Utils.json_utils import extract_valid_json, generate_gemini_prompt
import Utils.hf_models as hf

# Configure API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Function to evaluate the dataset with Gemini
def evaluate_dataset(conversations):
    """
    Evaluates a dataset of conversations using the Gemini API.

    This function iterates over a list of conversation entries, evaluates each conversation
    using the Gemini API, and collects the evaluation results.

    :param conversations: List of conversation entries. Each entry is expected to be a dictionary
                          with at least a "conversation" key containing the conversation text.
    :return: List of dictionaries containing evaluation results. Each dictionary includes
             "conversation_id" and "evaluation_scores".
    """

    # Run evaluation on all conversations
    evaluation_results = []
    # print(conversations)
    for entry in conversations:
        conversation_text = entry.get("conversation", "")
        if conversation_text:
            evaluation_data = evaluate_conversation_with_gemini(conversation_text)
            # print(evaluation_data)
            evaluation_results.append(
                {
                    "conversation_id": entry.get(
                        "conversation_id", len(evaluation_results) + 1
                    ),
                    "evaluation_scores": evaluation_data,
                }
            )

    return evaluation_results


# Function to evaluate conversation using Gemini with retry mechanism
def evaluate_conversation_with_gemini(conversation_text, max_retries=7, initial_wait=5):
    """
    Uses Gemini API to evaluate chatbot conversation quality and return structured scores.
    Retries API calls with exponential backoff if rate limit is hit.

    :param conversation_text: The chatbot conversation to evaluate.
    :param max_retries: Maximum number of retries if a request fails.
    :param initial_wait: Initial wait time in seconds before retrying.
    :return: Evaluated JSON response or default scores.
    """

    prompt = generate_gemini_prompt(conversation_text)  # Generate prompt
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "max_output_tokens": 500,
            "response_mime_type": "application/json",
        },
    )

    retries = 0
    wait_time = initial_wait  # Initial wait time (5 sec, can be adjusted)

    while retries < max_retries:
        try:
            response = model.generate_content(prompt)

            if response and hasattr(response, "text"):
                raw_output = response.text.strip()
                # print("\n RAW GEMINI OUTPUT:", raw_output)  # Debugging

                # Extract valid JSON
                evaluation_json = extract_valid_json(raw_output)
                if evaluation_json:
                    print("Successfully extracted JSON!")
                    return evaluation_json  # Return structured data

            print("Gemini returned non-JSON output. Using default scores.")
            return {
                "Relevance": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Coherence": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Factual Accuracy": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Bias & Toxicity": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Fluency": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Image Alignment": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
                "Creativity": {
                    "score": 5,
                    "explanation": "Evaluation uncertain due to lack of context.",
                },
            }

        except Exception as e:
            if "429" in str(e) or "quota" in str(e) or "exhausted" in str(e):
                print(
                    f"Rate limit exceeded! Retrying in {wait_time} seconds... ({retries + 1}/{max_retries})"
                )
                time.sleep(wait_time)  # Wait before retrying
                retries += 1
                wait_time *= 2  # Exponential backoff (5s → 10s → 20s → 40s...)
            else:
                print(f"Error in Gemini API call: {e}")
                break  # Exit loop for non-429 errors

    print("Maximum retries reached. Using default scores.")
    return {
        "Relevance": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Coherence": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Factual Accuracy": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Bias & Toxicity": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Fluency": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Image Alignment": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
        "Creativity": {
            "score": 5,
            "explanation": "Evaluation uncertain due to API error.",
        },
    }


def eval_dataset_hf_models(conversations):
    """
    Evaluates a dataset of conversations with evaluation functions using transformer models from HuggingFace.

    This function iterates over a list of conversation entries, evaluates each conversation
    using the evaluation functions from Utils.hf_models, and collects the evaluation results.

    :param conversations: List of conversation entries. Each entry is expected to be a dictionary
                          with at least a "conversation" key containing the conversation text.
    :return: List of dictionaries containing evaluation results. Each dictionary includes
             "conversation_id" and "evaluation_scores".
    """
    results = []
    past_responses = []  # Stores previous bot responses for creativity evaluation

    for idx, conversation_data in enumerate(conversations, 1):
        conversation_text = conversation_data.get("conversation", "")
        turns = conversation_text.split("\n\n")  # Split turns by double newlines

        # Extract all user prompts and bot responses
        user_prompts = [
            t.replace("HUMAN:", "").replace("[HUMAN]:", "").strip() 
            for t in turns if "HUMAN:" in t or "[HUMAN]:" in t
        ]
        bot_responses = [
            t.replace("BOT:", "").replace("[BOT]:", "").strip() 
            for t in turns if "BOT:" in t or "[BOT]:" in t
        ]

        # Extract images from JSON (if present)
        image_tag_mapping = conversation_data.get("image_tag_mapping", {})
        encoded_images = conversation_data.get("images", [])

        # Apply evaluations
        relevance_score, relevance_explanation = hf.evaluate_relevance(user_prompts, bot_responses)
        coherence_score, coherence_explanation = hf.evaluate_coherence(conversation_text)
        factual_accuracy_score, factual_accuracy_explanation = hf.evaluate_factual_accuracy(bot_responses)
        bias_toxicity_score, bias_toxicity_explanation = hf.evaluate_bias_toxicity(bot_responses)
        fluency_score, fluency_explanation = hf.evaluate_fluency(bot_responses)
        image_alignment_score, image_alignment_explanation = hf.evaluate_image_alignment(bot_responses, encoded_images, image_tag_mapping)
        creativity_score, creativity_explanation = hf.evaluate_creativity(bot_responses, past_responses)

        # Store past responses for creativity comparison
        past_responses.extend(bot_responses)

        # Store results
        results.append({
            "conversation_id": idx,
            "evaluation_scores": {
                "Relevance": {"score": relevance_score, "explanation": relevance_explanation},
                "Coherence": {"score": coherence_score, "explanation": coherence_explanation},
                "Factual Accuracy": {"score": factual_accuracy_score, "explanation": factual_accuracy_explanation},
                "Bias & Toxicity": {"score": bias_toxicity_score, "explanation": bias_toxicity_explanation},
                "Fluency": {"score": fluency_score, "explanation": fluency_explanation},
                "Image Alignment": {"score": image_alignment_score, "explanation": image_alignment_explanation},
                "Creativity": {"score": creativity_score, "explanation": creativity_explanation}
            }
        })
    
    return results

