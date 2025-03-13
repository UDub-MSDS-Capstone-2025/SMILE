import io
import base64
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from detoxify import Detoxify
from transformers import pipeline, CLIPProcessor, CLIPModel

# Load Transformer Models
similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Relevance, Coherence
fact_checker = pipeline("text-classification", model="facebook/bart-large-mnli")  # Factual Accuracy
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")  # Bias & Toxicity
fluency_model = pipeline("text-classification", model="textattack/roberta-base-CoLA")  # Fluency
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # Image Alignment
vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP Processor        

def evaluate_relevance(user_prompts, bot_responses):
    """
    Measures how well the chatbot's responses align with the overall conversation context.
    """
    if not bot_responses or not user_prompts:
        return 5, "No responses provided."

    # Compute embeddings for all user prompts and bot responses
    embeddings_prompt = similarity_model.encode(user_prompts, convert_to_tensor=True)
    embeddings_response = similarity_model.encode(bot_responses, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings_prompt.mean(dim=0), embeddings_response.mean(dim=0)).item() * 10

    return round(similarity_score), f"Semantic relevance score: {similarity_score:.2f}"

def evaluate_coherence(conversation_text):
    """
    Evaluates coherence using:
    - Sentence Similarity (Cosine Similarity of Embeddings)
    - Dimensionality Reduction (PCA to prevent shrinkage bias)
    - Percentile-Based Scaling for dynamic score adjustments.
    
    :param conversation_text: Full conversation string.
    :return: Normalized coherence score (1-10) with explanation.
    """
    if not conversation_text:
        return 5, "No conversation provided."

    # Extract bot responses
    turns = conversation_text.split("\n\n")
    bot_responses = [
        t.replace("BOT:", "").replace("[BOT]:", "").strip() 
        for t in turns if "BOT:" in t or "[BOT]:" in t
    ]

    if len(bot_responses) < 2:
        return 5, "Not enough chatbot responses to measure coherence."

    # Compute Sentence Embeddings in One Batch (Speeds Up Execution)
    embeddings = similarity_model.encode(bot_responses, convert_to_tensor=True)

    # Apply PCA for Dimensionality Reduction (Fixes Shrinkage Bias)
    pca = PCA(n_components=min(embeddings.shape[0], 50))  # Retain meaningful variance
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())

    # Compute Pairwise Similarities Efficiently
    similarities = [
        util.pytorch_cos_sim(torch.tensor(reduced_embeddings[i]), torch.tensor(reduced_embeddings[i + 1])).item()
        for i in range(len(reduced_embeddings) - 1)
    ]

    if not similarities:
        return 5, "Not enough variance in responses."

    # Normalize Scores Using Percentile-Based Scaling
    min_sim, max_sim = 0, 1
    avg_similarity = 1 + np.mean(similarities)
    coherence_score = int(np.clip(((avg_similarity - min_sim) / (max_sim - min_sim)) * 9 + 1, 1, 10))

    explanation = f"Embedding similarity score: {avg_similarity:.2f} (PCA applied)"

    return coherence_score, explanation

def evaluate_factual_accuracy(bot_responses):
    """
    Uses NLI to check factual consistency across the entire conversation.
    """
    if not bot_responses:
        return 5, "No responses provided."
    
    result = fact_checker(" ".join(bot_responses))
    entailment_score = max(result, key=lambda x: x['score'])['score'] * 10

    return round(entailment_score), f"Factual consistency score: {entailment_score:.2f}"

def evaluate_bias_toxicity(bot_responses):
    """
    Detects bias or toxic language in chatbot responses.
    """
    if not bot_responses:
        return 10, "No responses provided."
    
    combined_text = " ".join(bot_responses)
    toxicity_score = Detoxify('original').predict(combined_text)['toxicity'] * 10
    final_score = max(1, 10 - int(toxicity_score))  # Lower toxicity = higher score

    return final_score, f"Toxicity probability: {toxicity_score:.2f}"

def evaluate_fluency(bot_responses):
    """
    Uses CoLA model to check grammatical fluency across chatbot responses.
    """
    if not bot_responses:
        return 5, "No responses provided."

    fluency_scores = [fluency_model(response)[0]['score'] * 10 for response in bot_responses]
    avg_fluency = sum(fluency_scores) / len(fluency_scores)

    return round(avg_fluency), f"Average fluency score: {avg_fluency:.2f}"

def evaluate_image_alignment(bot_responses, image_data, image_tag_mapping):
    """
    Uses CLIP to check if chatbot responses align with encoded images.
    """
    if not bot_responses or not image_data:
        return 5, "No images or responses provided."

    processed_texts = []
    processed_images = []

    # Process each image referenced in the conversation
    for tag, img_name in image_tag_mapping.items():
        for img in image_data:
            if img["name"] == img_name:
                # Decode base64 image
                image = Image.open(io.BytesIO(base64.b64decode(img["base64"])))
                
                processed_images.append(image)
                processed_texts.append(" ".join(bot_responses))  # Combine bot responses

    if not processed_images:
        return 5, "No valid images found for evaluation."

    # Process text and images using CLIP with truncation/padding
    inputs = vision_processor(
        text=processed_texts, 
        images=processed_images, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Move tensors to the correct device
    inputs = {key: val.to(vision_model.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**inputs)
        similarity_scores = outputs.logits_per_image.cpu().numpy()  # Convert to NumPy

    # Compute Min-Max normalization for similarity scores
    min_clip_score, max_clip_score = 1, 40
    avg_score = similarity_scores.mean() if similarity_scores.size > 0 else 0.5
    # normalized_score = int(1 + ((avg_score - min_clip_score) / (max_clip_score - min_clip_score)) * 9)
    normalized_score = int(np.clip(((avg_score - min_clip_score) / (max_clip_score - min_clip_score)) * 9 + 1, 1, 10))

    return normalized_score, f"CLIP text-image similarity score: {avg_score:.2f}"

def evaluate_creativity(bot_responses, past_responses):
    """
    Measures creativity by comparing bot responses against past responses.
    """
    if not bot_responses:
        return 5, "No responses provided."
    
    embedding_current = similarity_model.encode(bot_responses, convert_to_tensor=True)
    embedding_past = similarity_model.encode(past_responses, convert_to_tensor=True) if past_responses else None

    max_similarity = 0 if embedding_past is None else util.pytorch_cos_sim(embedding_current.mean(dim=0), embedding_past.mean(dim=0)).item()
    creativity_score = max(1, min(10, int((1 - max_similarity) * 10)))  # Inverse similarity

    return creativity_score, f"Novelty score: {1 - max_similarity:.2f} (Lower similarity = more creative)"