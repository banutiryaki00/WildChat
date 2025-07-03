WildChat Dataset: Intent-Driven Topic Analysis for Purchase Behavior

This repository contains an analysis of the WildChat dataset from Hugging Face, focusing on understanding user intent patterns and identifying genuine purchase behavior in AI conversations. The analysis combines traditional topic modeling with AI-powered intent classification.

Dataset Information
Source: WildChat Dataset by Allen Institute for AI
Access: https://huggingface.co/datasets/allenai/WildChat

Classes Implemented:
•IntentClassifier: Rule-based intent classification using regex patterns
•TextPreprocessor: Text cleaning and preprocessing for topic modeling

Intent Categories:
•Information_Seeking: Questions and explanations (7.2%)
•Creative_Assistance: Content creation requests (26.5%)
•Coding_Programming: Technical development queries (21.1%)
•Decision_Support: Choice and recommendation requests (3.4%)
•Problem_Solving: Troubleshooting and fixes (1.3%)
•Learning_Skill_Development: Educational content (1.3%)
•Text_Editing_Formatting: Writing improvement (1.5%)
•Math_Calculations: Mathematical problems (1.5%)
•Roleplay_Character: Character and narrative content (3.5%)
•Translation_Language: Language translation (0.7%)
•General_Conversation: Social interactions (2.0%)

2.Topic Modeling Pipeline
Functions:
•extract_first_user_message(): Extracts initial user queries from conversations
•extract_conversation_info(): Processes conversation metadata
•extract_top_words(): Identifies key words for each topic
•create_topic_name(): Generates topic labels

3. Topic Modeling Process:
3.1.Text Preprocessing: Cleaning, tokenization, stop word removal
3.2.Vectorization: TF-IDF and Count vectorization for LDA input
3.3.LDA Modeling: Latent Dirichlet Allocation with adaptive topic numbers
3.4.Topic Extraction: Identification and naming of discovered topics
3.5.Quality Assessment: Perplexity scoring and coherence evaluation

4. Purchase Intent Analysis

AI-Powered Classification:
•Model: GPT-3.5-Turbo
•Input: User conversation text
•Output: JSON with purchase insights
Classification Schema:
{
  "purchase_intent_score": "1-10 scale",
  "purchase_category": "electronics|software|automotive|apparel|...",
  "specific_product": "exact product or 'none'",
  "price_range": "under_100|100_500|500_2000|over_2000|not_mentioned",
  "decision_stage": "research|compare|ready_to_buy|price_shopping",
  "key_decision_factors": ["price", "features", "brand", "reviews", "quality"]
}
Analysis Functions:
•analyze_purchase_conversation(): Single conversation analysis via OpenAI API
•get_purchase_conversations(): Extracts purchase-related conversations using regex patterns


Technical Requirements
datasets
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
gensim
nltk
spacy
sentence-transformers
umap-learn
hdbscan




