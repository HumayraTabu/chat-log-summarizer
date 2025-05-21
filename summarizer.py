import os
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def parse_chat_log(file_path):
    user_messages = []
    ai_messages = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("User:"):
                user_messages.append(line[len("User:"):].strip())
            elif line.startswith("AI:"):
                ai_messages.append(line[len("AI:"):].strip())

    return user_messages, ai_messages


def print_message_statistics(user_messages, ai_messages):
    total_messages = len(user_messages) + len(ai_messages)
    print(f"Total Messages: {total_messages}")
    print(f"User Messages: {len(user_messages)}")
    print(f"AI Messages: {len(ai_messages)}")


def tokenize_and_filter(messages):
    text = ' '.join(messages).lower()
    words = re.findall(r'\b\w+\b', text)
    return [word for word in words if word not in stop_words]


def extract_keywords_tfidf(messages, top_n=5):
    corpus = [' '.join(messages)]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:top_n]]


def generate_summary(user_msgs, ai_msgs, filename, use_tfidf=True):
    total_exchanges = len(user_msgs) + len(ai_msgs)
    all_msgs = user_msgs + ai_msgs

    # Keyword extraction
    keywords = extract_keywords_tfidf(all_msgs) if use_tfidf else tokenize_and_filter(all_msgs)
    keywords = keywords[:5]

    # Topic heuristics
    if keywords:
        topic = f"The conversation mainly revolved around '{keywords[0]}'"
    else:
        topic = "The conversation covered general topics"

    # Summary Output
    print(f"\nSummary for '{filename}'")
    print(f"- The conversation had {total_exchanges} exchanges.")
    print(f"- {topic}")
    print(f"- Most common keywords: {', '.join(keywords)}")


def analyze_and_summarize_folder(folder_path, use_tfidf=True):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"\n{filename} Results")
            user_msgs, ai_msgs = parse_chat_log(file_path)

            print("User Messages:")
            for msg in user_msgs:
                print("-", msg)

            print("\nAI Messages:")
            for msg in ai_msgs:
                print("-", msg)

            print_message_statistics(user_msgs, ai_msgs)

            all_msgs = user_msgs + ai_msgs
            top_keywords = extract_keywords_tfidf(all_msgs)
            print("\nTop 5 Keywords: " + ", ".join(top_keywords))

            # Print summary
            generate_summary(user_msgs, ai_msgs, filename, use_tfidf)


# --- Main block ---
if __name__ == "__main__":
    folder_path = "chat_logs"  # Put your folder name here
    analyze_and_summarize_folder(folder_path, use_tfidf=True)
