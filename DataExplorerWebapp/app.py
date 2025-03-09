import streamlit as st
import pandas as pd
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gdown
import os


# ----------------- APP CONFIG -----------------
st.set_page_config(page_title="Synthetic Data Generation for Multi-modal LLMs", layout="wide")

# # Force Streamlit to apply a higher max message size
# st.set_option("server.maxMessageSize", 5000)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Dataset Explorer"])

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.title("🧠 Synthetic Data Generation for Multi-modal LLMs")
    st.markdown("""
    ## Welcome to the Synthetic Data Generation Dashboard!  
    This project focuses on generating high-quality **multi-modal datasets** using **Gemini AI** and evaluating chatbot responses with **Gemini AI** based on 3H (Honesty, Helpfulness and Harmlessness) parameters.  

    ### 📌 **Project Objectives**
    - Generate synthetic **human-bot conversations** based on **text and images**.
    - Ensure **ethical AI** by preventing biased, toxic, or identifiable personal information.
    - **Evaluate** chatbot responses using **multiple LLM models** to assess quality.

    ### 🔍 **Methodology**
    1. **Synthetic Data Generation**:  
       - Uses **Gemini AI** to generate human-bot conversations.
       - Includes **multi-turn dialogues** with references to images.
    2. **Dataset Evaluation**:  
       - Uses **Gemini** to provide **7 evaluation scores** per conversation:
         - **Relevance, Coherence, Factual Accuracy, Bias, Fluency, Image Alignment, Creativity**.
    3. **Dataset Explorer & Visualization**:  
       - Interactive filtering and visualization of scores.
       - Image thumbnail previews for conversations.

    ### 🚀 **Key Features**
    - 📊 **Dataset Filtering & Score Visualization**
    - 🖼️ **Image Previews & Mapping**
    - 📥 **Download Filtered Dataset**
    """)


    st.info("🔄 Use the sidebar to navigate to the **Dataset Explorer**!")

# ----------------- DATASET EXPLORER PAGE -----------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

    # Sidebar: Dataset Selection
    st.sidebar.header("📂 Select Dataset Category")
    dataset_category = st.sidebar.selectbox("Choose Dataset Type", ["Anime", "Celeb", "Meme", "Clustered", "Combined"])

    # Define dataset file paths based on selection
    dataset_paths = {
        # "Anime": "../Final_Datasets/anime.json",
        # "Celeb": "../Final_Datasets/celeb.json",
        # "Meme": "../Final_Datasets/meme.json",
        # "Clustered": "../Final_Datasets/clustering.json",
        # "Combined": "../Final_Datasets/combined_folder.json"
        "Anime": "18EA2dgaMPxuJ1VGeYYgfp9TXXyjmLuIK",
        "Celeb": "1zhmP7QrD_ZZN8Mm5ekHZMPyVmwN877D_",
        "Meme": "1SzE0BKiOo7xV7R7D1Vr30pnoKTcyoXqu",
        "Clustered": "1Dz25PN-54OYPD0ZZ9fb9apGC40Z0bK6-",
        "Combined": "196X5cOhQu-KRyyUHxAGyNynTu38oR-Jh"
    }

    evaluation_paths = {
        # "Anime": "../Evaluation_result/clustering_part1_200_baisakhi_evaluation_results0224.json",
        # "Celeb": "../Evaluation_result/clustering_part1_200_baisakhi_evaluation_results0224.json",
        # "Meme": "../Evaluation_result/clustering_part1_200_baisakhi_evaluation_results0224.json",
        # "Clustered": "../Evaluation_result/clustering_part1_200_baisakhi_evaluation_results0224.json",
        # "Combined": "../Evaluation_result/clustering_part1_200_baisakhi_evaluation_results0224.json"
        "Anime": "1mwxYkfKN6ACy-zr-xPlFDhe2YCqmC9oU",
        "Celeb": "1Srcb3wWA1khv2ZQMSt8oRMqSjTmiLlqz",
        "Meme": "1HZtLo8iJo2rz32eJ8lVBYiZ6zo3H6C4W",
        "Clustered": "154nbfikh9VuPnER-XNxoo3ureVNKF-0o",
        "Combined": "1bVFfXtQBCfku3R3JZpAPM76nEpimF9AD"
    }

    # ----------------- DATA LOADING FUNCTIONS -----------------
    @st.cache_data
    # def load_conversation_data(json_file):
    #     with open(json_file, "r") as file:
    #         return pd.json_normalize(json.load(file), sep="_")

    @st.cache_data
    def download_from_gdrive(file_id):
        """Downloads a file from Google Drive and returns its local path."""
        url = f"https://drive.google.com/uc?id={file_id}"
        output = f"temp_{file_id}.json"  # Unique temp filename
        if not os.path.exists(output):  # Download only if not already present
            gdown.download(url, output, quiet=False)
        return output

    def load_conversation_data(file_id, chunk_size=5000):
        """
        Lazily loads large conversation datasets in chunks to prevent memory overflow.
        Returns only the first chunk.
        """
        # Read from local
        # with open(json_file, "r") as file:
        #     data = json.load(file)  # Load JSON normally
    
        # df = pd.json_normalize(data, sep="_")  # Convert JSON to DataFrame
        # return df.iloc[:chunk_size]  # Load only the first `chunk_size` rows

        # Read from google drive
        json_file = download_from_gdrive(file_id)
    
        with open(json_file, "r") as file:
            data = json.load(file)  # Load JSON normally

        df = pd.json_normalize(data, sep="_")  # Convert JSON to DataFrame
        # Convert list columns to strings for caching (Fixes Pandas Hashing Issue)
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
        return df.iloc[:chunk_size]  # Load only the first chunk
        

    @st.cache_data
    def load_evaluation_data(file_id):
        #Read from local
        # with open(json_file, "r") as file:
        #     data = json.load(file)
        
        # for entry in data:
        #     for key, value in entry["evaluation_scores"].items():
        #         entry["evaluation_scores"][key] = value["score"]  # Keep only scores

        # read from google drive
        """Loads evaluation data and keeps only the scores."""
        json_file = download_from_gdrive(file_id)

        with open(json_file, "r") as file:
            data = json.load(file)
        
        for entry in data:
            for key, value in entry["evaluation_scores"].items():
                entry["evaluation_scores"][key] = value["score"]  # Keep only scores

        return pd.json_normalize(data, sep="_")

    @st.cache_data
    def convert_df_to_json(df):
        return df.to_json(orient="records", indent=4)

    # Function to decode base64 image
    def decode_base64_image(encoded_string):
        """Decodes a base64 image and returns an HTML image tag."""
        return f'<img src="data:image/png;base64,{encoded_string}" style="width:50px;height:50px;" />' 
    
    
    # Load selected dataset
    conversation_data = load_conversation_data(dataset_paths[dataset_category])
    evaluation_data = load_evaluation_data(evaluation_paths[dataset_category])

    # Merge evaluation scores into conversation data
    merged_data = conversation_data.merge(evaluation_data, on="conversation_id", how="left")

    # ----------------- FILTERING OPTIONS -----------------
    st.sidebar.header("🔍 Filter Options")

    if "images" in merged_data.columns:
        image_counts = merged_data['images'].apply(len).unique()
        selected_image_count = st.sidebar.multiselect("Select Number of Images", image_counts, default=image_counts)

    score_columns = [col for col in evaluation_data.columns if "_score" in col]

    selected_score = None
    if score_columns:
        selected_score = st.sidebar.selectbox("Filter by Score Metric", score_columns)
        min_score, max_score = st.sidebar.slider("Select Score Range", 0, 10, (5, 10))
    else:
        st.sidebar.error("⚠️ No evaluation score columns found!")

    search_text = st.sidebar.text_input("Search in Conversation")

    # Apply Filters
    filtered_conversations = merged_data.copy()

    if "images" in merged_data.columns and selected_image_count:
        filtered_conversations = filtered_conversations[filtered_conversations['images'].apply(len).isin(selected_image_count)]

    if selected_score and selected_score in merged_data.columns:
        filtered_conversations = filtered_conversations[filtered_conversations[selected_score].between(min_score, max_score)]

    if search_text and "conversation" in merged_data.columns:
        filtered_conversations = filtered_conversations[filtered_conversations["conversation"].str.contains(search_text, case=False, na=False)]

    # ----------------- DISPLAY FILTERED DATA -----------------
    # st.subheader("📊 Filtered Conversations")

    # if not filtered_conversations.empty:
    #     json_data = convert_df_to_json(filtered_conversations)
    #     st.download_button("📥 Download Filtered Data (JSON)", data=json_data, file_name="filtered_dataset.json", mime="application/json")
    #     st.dataframe(filtered_conversations)
    # else:
    #     st.warning("⚠️ No data matches your filters.")

    # # ----------------- VISUALIZATIONS -----------------
    # if not filtered_conversations.empty:
    #     avg_scores = filtered_conversations[score_columns].mean().reset_index()
    #     avg_scores.columns = ["Metric", "Average Score"]

    #     avg_scores["Metric"] = avg_scores["Metric"].str.replace("evaluation_scores_", "").str.replace("_score", "").str.replace("_", " ").str.title()

    #     st.subheader("📊 Average Scores by Metric (Filtered Data)")
    #     fig = px.bar(avg_scores, x="Metric", y="Average Score", color="Metric", text="Average Score")
    #     st.plotly_chart(fig)

    #     st.subheader("🔥 Heatmap of Evaluation Scores")
    #     plt.figure(figsize=(10, 5))
    #     sns.heatmap(filtered_conversations[score_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    #     st.pyplot(plt)

    # ✅ Define evaluation score columns
    score_columns = [
        "evaluation_scores_Relevance",
        "evaluation_scores_Coherence",
        "evaluation_scores_Factual Accuracy",
        "evaluation_scores_Bias & Toxicity",
        "evaluation_scores_Fluency",
        "evaluation_scores_Image Alignment",
        "evaluation_scores_Creativity"
    ]

    # ✅ Ensure filtered_data is not empty before calculations
    if not filtered_conversations.empty:
        # ✅ Compute average scores
        avg_scores = filtered_conversations[score_columns].mean().reset_index()
        avg_scores.columns = ["Metric", "Average Score"]  # Rename columns

        # ✅ Rename metrics for better readability
        clean_labels = {
            "evaluation_scores_Relevance": "Relevance",
            "evaluation_scores_Coherence": "Coherence",
            "evaluation_scores_Factual Accuracy": "Factual Accuracy",
            "evaluation_scores_Bias & Toxicity": "Bias & Toxicity",
            "evaluation_scores_Fluency": "Fluency",
            "evaluation_scores_Image Alignment": "Image Alignment",
            "evaluation_scores_Creativity": "Creativity"
        }
        avg_scores["Metric"] = avg_scores["Metric"].replace(clean_labels)

        # ✅ Re-plot bar chart with updated labels
        st.subheader("📊 Average Scores by Metric (Filtered Data)")
        fig = px.bar(avg_scores, x="Metric", y="Average Score", color="Metric", text="Average Score")
        fig.update_layout(xaxis_title="Evaluation Metric", yaxis_title="Average Score")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ No data available after filtering. Adjust filters to see results.")


    # Show Filtered Dataset with Image Thumbnails and Image-to-Tag Mapping
    st.subheader("📊 Filtered Conversations")




    if not filtered_conversations.empty:
        json_data = convert_df_to_json(filtered_conversations)
        st.download_button(
            label="📥 Download Filtered Data (JSON)",
            data=json_data,
            file_name="filtered_dataset.json",
            mime="application/json",
        )

        for index, row in filtered_conversations.iterrows():
            st.markdown(f"### **Conversation ID: {row['conversation_id']}**")

            # Image-to-Tag Mapping
            st.markdown("**📷 Image-to-Tag Mapping:**")
            image_mappings = {}
            for idx, img_data in enumerate(row["images"]):
                img_name = img_data["name"]
                img_tag = f"<img_{idx+1}>"
                image_mappings[img_tag] = img_name
            
            st.json(image_mappings)  # Display mapping

            # Show Images as Thumbnails
            st.markdown("**🖼️ Images Used:**")
            image_html = ""
            for img in row["images"]:
                image_html += decode_base64_image(img["base64"]) + " "
            
            st.markdown(image_html, unsafe_allow_html=True)  # Render images inline
            
            # Show Conversation
            st.markdown(f"**💬 Conversation:** {row['conversation']}")

            # Show Scores
            st.markdown("**📊 Evaluation Scores:**")
            scores = {key: row[key] for key in score_columns if key in row}
            st.json(scores)

            st.divider()  # Add a separator between conversations

    else:
        st.warning("⚠️ No data matches your filters.")


