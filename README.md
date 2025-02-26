# Crop Chat-Bot

This project is a conversational chatbot developed to assist farmers and agricultural advisors in identifying and managing crop diseases. The bot leverages deep learning and natural language processing (NLP) techniques, specifically using **BERT embeddings**, to comprehend user queries about crop diseases and provide helpful, context-aware guidance. By employing a pre-trained and fine-tuned model, the chatbot improves its understanding of agricultural terminology and scenarios, providing users with accurate responses.

## Features

- **BERT-based Embeddings**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) embeddings to interpret and respond accurately to user queries. BERT allows the model to understand the context of words in a sentence, leading to more accurate responses.
- **Pre-trained and Fine-tuned Model**: A pre-trained Transformer model (e.g., GPT-2) is fine-tuned on a curated dataset of disease-related conversations, enabling the bot to recognize and respond appropriately to agricultural-related inquiries.
- **Spell Correction**: Implements an edit distance algorithm to automatically detect and correct minor spelling errors in user queries, ensuring that the model still understands the user’s intent.
- **Streamlit Interface**: The bot features an interactive and user-friendly interface powered by Streamlit, allowing farmers and advisors to easily access information on crop diseases and solutions.

## Dataset Creation

To train the model to understand agricultural queries, a conversational dataset was created, containing pairs of user inputs and corresponding responses. The dataset includes various intents, such as:
- **Crop Disease Identification**: Queries about specific diseases affecting crops (e.g., “What is the cause of yellow leaves on tomato plants?”).
- **Treatment Advice**: Queries asking for remedies for different crop diseases.
- **General Agricultural Queries**: Queries related to crop growth, environmental conditions, or pest management.

The dataset was organized in a JSON format with each intent tagged appropriately. This dataset serves as the foundation for fine-tuning the model, allowing it to learn from examples and respond accurately to real-world queries.

## Preprocessing

The preprocessing steps for the dataset include:

- **Tokenization**: Tokenizing the input text helps break down sentences into smaller components (tokens), such as words or sub-words, which are easier for the model to understand.
- **Lemmatization**: Reduces words to their base or root form (e.g., “running” becomes “run”), which reduces vocabulary complexity and helps improve model performance.
- **Normalization**: Removes unnecessary characters (such as punctuation and special symbols), converts all text to lowercase, and corrects minor spelling errors using an edit distance algorithm.
- **Dataset Formatting**: Ensures that the data is correctly structured in JSON format, with each user query paired with the corresponding response. This format is essential for training and fine-tuning the model.

## Model Training

### Embedding Generation:
- **BERT Embeddings**: BERT embeddings were used to capture sentence-level meaning and context, allowing the model to understand complex and ambiguous queries. By leveraging BERT's pre-trained knowledge, the model can achieve better performance with fewer training examples.

### Fine-tuning the Model:
- A pre-trained Transformer model, such as **GPT-2**, was fine-tuned on the agricultural dataset to enhance its ability to respond to crop disease-related queries. Fine-tuning involves adjusting the weights of the pre-trained model using the task-specific dataset to adapt it to the context of the crop disease domain.
  
### Hyperparameter Tuning:
- Hyperparameters such as **learning rate**, **batch size**, **epochs**, and the **number of layers** were optimized to improve model performance.
  - **Learning Rate**: Tuned within a range to find the best rate at which the model updates its weights during training. A lower learning rate prevents overshooting of the optimal weight values, but too small of a learning rate could result in slow convergence.
  - **Batch Size**: Adjusted to control how many samples the model processes before updating its weights. Larger batch sizes can speed up training, but might also introduce noise.
  - **Epochs**: Set to an optimal value to ensure the model trained for a sufficient amount of time without overfitting. Early stopping was also implemented to prevent overfitting by stopping training when validation loss no longer improves.
  - **Number of Layers/Units**: The number of layers in the neural network and the units per layer were fine-tuned to balance model capacity with computational efficiency.

## Evaluation Metrics

To evaluate the performance of the Crop Chat-Bot, two key metrics were used:

1. **BLEU Score (Bilingual Evaluation Understudy Score)**:
   - The **BLEU score** is a metric commonly used to evaluate the quality of machine-generated text, particularly in machine translation tasks. In the case of the Crop Chat-Bot, BLEU measures how closely the generated responses match the reference responses in the dataset.
   - The BLEU score ranges from 0 to 1, with 1 indicating perfect similarity between the model's output and the reference response. A higher BLEU score indicates that the model is providing responses that are closer to the ideal response.
   - A score of **0.7 or above** typically indicates strong performance, but in cases of conversational systems, some variations in wording may be acceptable.
   - My model had a bleu score of 0.88 which is very good

2. **F1 Score**:
   - The **F1 score** is a harmonic mean of precision and recall and is especially useful when there is an imbalance in the class distribution (i.e., certain diseases or queries might be more common than others).
     - **Precision**: Measures how many of the model’s responses are relevant (i.e., the percentage of relevant responses out of all responses generated).
     - **Recall**: Measures how many relevant responses the model actually generated (i.e., the percentage of relevant responses out of all the possible correct responses).
   - F1 score is particularly valuable when you need a balance between precision and recall, and is used to assess the chatbot’s ability to provide accurate and relevant answers across all disease queries.
   - I had an F1 score of 0.7.. so that is also very good!!

### Model Accuracy:
- The **accuracy** of the model was also computed, which represents the proportion of correct responses (i.e., the number of times the bot’s response matches the expected response) compared to the total number of queries. While accuracy is useful, it does not account for the nuances of different types of responses, which is why the F1 score and BLEU score provide additional insights into model performance.
- My model reached an accuracy of 94%.. which means my model is doing pretty well!

## Deployment

The Crop Chat-Bot is deployed via **Streamlit**, providing an easy-to-use interface for farmers and advisors to interact with the model. Users can type in their queries about crop diseases, and the chatbot will respond with relevant information or advice.

- **Web Demo**: The chatbot is hosted on the following platform for easy access:
  - [Crop Chat-Bot Demo](https://thecropbot.streamlit.app/)

- **Project Archive**: The project code and demo can be found in the following Google Drive link:
  - [Project Google Drive](#)

## Future Work

- **Expansion of Dataset**: The model can be enhanced by expanding the dataset with more diverse queries and responses to cover additional crop diseases and agricultural scenarios.
- **Multi-Lingual Support**: Adding multi-lingual support to the bot to cater to users from different regions and with different language preferences.
- **Real-time Integration**: The chatbot can be integrated with real-time crop disease monitoring tools or sensors, providing users with up-to-date information on disease outbreaks in their area.

---
