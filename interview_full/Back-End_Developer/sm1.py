import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(file_path):
    """Load the JSON dataset from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def parse_context(context):
    """
    Extract Q&A pairs from a context string.
    Uses a regex pattern to capture questions and answers in the format:
    Q<number>: <question text>
    A<number>: <answer text>
    """
    pattern = r"Q\d+: (.*?)\nA\d+: (.*?)(?=\n\n|$)"
    matches = re.findall(pattern, context, re.DOTALL)
    qa_pairs = [{'question': q.strip(), 'answer': a.strip()} for q, a in matches]
    return qa_pairs

def parse_dataset(data):
    """
    Parse the dataset by iterating over each paragraph in the JSON file
    and extracting all Q&A pairs from the context.
    """
    qa_pairs = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            qa_pairs.extend(parse_context(context))
    return qa_pairs

def train_chatbot(qa_pairs):
    """Prepare the chatbot: vectorize questions and train a nearest neighbors model."""
    questions = [qa['question'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(question_vectors)
    return vectorizer, nn_model, answers

def chatbot_response(user_input, vectorizer, nn_model, answers):
    """Return the answer that best matches the user's input."""
    user_vector = vectorizer.transform([user_input])
    _, idx = nn_model.kneighbors(user_vector)
    return answers[idx[0][0]]

def evaluate_answer(user_answer, correct_answer, vectorizer):
    """
    Evaluate the user's answer by computing the cosine similarity between
    the vectorized user answer and the correct answer.
    """
    user_vector = vectorizer.transform([user_answer])
    correct_vector = vectorizer.transform([correct_answer])
    similarity = cosine_similarity(user_vector, correct_vector)[0][0]
    return similarity

def main():
    file_path = 'interview_full\Back-End_Developer\Back-End_Developer_high.json'  # Ensure this path points to your JSON file
    data = load_dataset(file_path)
    qa_pairs = parse_dataset(data)
    
    # Train the chatbot model
    vectorizer, nn_model, answers = train_chatbot(qa_pairs)
    
    print("Chatbot: Hi! Let's start the Q&A session.")
    correct_count = 0
    
    # Randomly choose 5 questions from the dataset for the session
    questions = [qa['question'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    random_indices = random.sample(range(len(questions)), 5)
    
    for idx in random_indices:
        question = questions[idx]
        correct_answer = answers[idx]
        print("Chatbot:", question)
        user_answer = input("You: ")
        similarity_score = evaluate_answer(user_answer, correct_answer, vectorizer)
        
        if similarity_score > 0.5:
            print("Chatbot: Good job! Your answer is quite accurate.")
            correct_count += 1
        else:
            print("Chatbot: The correct answer is:", correct_answer)
        print()
        
    print(f"Chatbot: You answered {correct_count}/5 questions correctly.")
    print("Chatbot: Thank you for participating!")

if __name__ == "__main__":
    main()
