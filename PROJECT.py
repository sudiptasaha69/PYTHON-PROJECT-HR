import nltk
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from fpdf import FPDF
from collections import Counter
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Load text file
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Summarize text using frequency method
def summarize_text(text, max_sentences=5):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english") + list(string.punctuation))

    word_freq = Counter([w for w in words if w not in stop_words])

    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_freq[word]
                else:
                    sentence_scores[sent] += word_freq[word]

    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    return ' '.join(sorted_sentences[:max_sentences])

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 to 1

# Word frequency plot
def plot_word_freq(text, top_n=10):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    filtered_words = [w for w in words if w not in stop_words]
    freq = Counter(filtered_words).most_common(top_n)

    df = pd.DataFrame(freq, columns=["Word", "Frequency"])
    df.plot(kind='bar', x='Word', y='Frequency', legend=False)
    plt.title("Top Word Frequencies")
    plt.tight_layout()
    plt.savefig("word_freq.png")
    plt.close()

# PDF Report
def create_pdf(summary, sentiment_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Book Summary and Sentiment Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}\n")
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    pdf.cell(0, 10, f"Sentiment Score: {sentiment_score:.2f} ({sentiment})", ln=True)

    pdf.image("word_freq.png", w=170)
    pdf.output("summary_report.pdf")

# Main Program
def main():
    filepath = input("Enter path to your book (.txt file): ")
    text = load_text(filepath)

    print("[✓] Summarizing text...")
    summary = summarize_text(text)

    print("[✓] Analyzing sentiment...")
    sentiment = analyze_sentiment(summary)

    print("[✓] Plotting word frequencies...")
    plot_word_freq(text)

    print("[✓] Generating PDF report...")
    create_pdf(summary, sentiment)

    print("[✓] Done! Report saved as summary_report.pdf")

if __name__ == "__main__":
    main()
