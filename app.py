from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Load intents from JSON file
with open('intents.json', 'r') as file:
    data = json.load(file)

intents_data = data.get("intents", [])
offensive_words = data.get("offensive_words", [])

# Preprocess the intents data
processed_data = []
intent_tags = {}

# NLTK setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

for idx, intent in enumerate(intents_data):
    examples = intent.get('patterns', [])  # get() handle missing 'patterns' key

    # Process each example
    for example in examples:
        # Tokenization, lowercasing, removing stop words, stemming, and removing punctuation
        tokens = word_tokenize(example.lower())
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric characters

        # Join tokens back into a sentence
        processed_example = ' '.join(tokens)

        # Store the processed example and its corresponding intent tag
        processed_data.append(processed_example)
        intent_tags[len(processed_data) - 1] = intent.get('tag', '')  # Provide a default value

# Create a bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_data)

# User-specific warning and block dictionary
user_warnings = {}

# Rule-based intent matching with offensive word detection
def rule_based_match(user_input):
    global user_warnings

    # Check for offensive words
    if any(word in user_input.lower() for word in offensive_words):
        user_id = hash(user_input)  # A simple way to identify users (replace with a more robust method)
        user_warnings[user_id] = user_warnings.get(user_id, 0) + 1

        # Check if the user has exceeded the warning threshold
        if user_warnings[user_id] >= 3:
            # Ban the user for the specified duration
            ban_expiration_time = datetime.now() + timedelta(seconds=10)
            user_warnings[user_id] = ban_expiration_time

            return "Sorry, but you have been temporarily banned from using the chatbot for the next hour due to multiple violations of community guidelines."

        return "Warning: The use of offensive language is not allowed. Please refrain from using inappropriate words."

    # Tokenize, lowercase, remove stop words, stem, and remove punctuation
    user_tokens = word_tokenize(user_input.lower())
    user_tokens = [token for token in user_tokens if token not in stop_words]
    user_tokens = [stemmer.stem(token) for token in user_tokens]
    user_tokens = [token for token in user_tokens if token.isalnum()]  # Remove non-alphanumeric characters
    processed_user_input = ' '.join(user_tokens)

    # Transform the processed user input using the vectorizer
    user_input_bow = vectorizer.transform([processed_user_input])

    # Find the closest match using a simple rule (cosine similarity)
    similarity_scores = (X @ user_input_bow.T).toarray().flatten()

    # Set a threshold for cosine similarity
    threshold = 0.2
    if max(similarity_scores) >= threshold:
        best_match_index = similarity_scores.argmax()
        return intent_tags.get(best_match_index)

    return None

def get_intent(tag):
    # Implement this function to get the intent based on the tag
    return next((intent for intent in intents_data if intent.get('tag', '') == tag), None)

def get_response(intent):
    # Implement this function to get the response based on the intent
    if intent is not None:
        responses = intent.get('responses', [])
        # Replace newline characters with HTML line break tags
        responses = [response.replace('\n', '<br>') for response in responses]
        return responses
    return None

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('user_input')

    # Perform rule-based matching
    matched_intent_tag = rule_based_match(user_input)

    if matched_intent_tag is not None:
        # Check for offensive words
        if any(word in user_input.lower() for word in offensive_words):
            response = "Warning! The use of offensive language is not allowed. Please refrain from using inappropriate words."
        else:
            # Find the matched intent
            matched_intent = get_intent(matched_intent_tag)

            # Get the responses
            responses = get_response(matched_intent)

            # Combine responses into a single string
            response = "<br>".join(responses) if responses else "I'm sorry, I couldn't find the information you're looking for."

    else:
        response = "I'm sorry, I couldn't find the information you're looking for. Please check your query and try again. If you need further assistance, you can contact our support team."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
