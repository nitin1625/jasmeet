import googleapiclient.discovery 
import re
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify,stream_with_context
from flask_cors import CORS
load_dotenv()

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

def get_youtube_comments(video_url, api_key):
    video_id = video_url.split('v=')[1].split('&')[0]

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=300
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment)

    return comments

def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        comment = re.sub(r"http\S+|www\S+|https\S+|@\S+", '', comment, flags=re.MULTILINE)
        comment = re.sub(r'\s+', ' ', comment)  
        comment = re.sub(r'[^\w\s]', '', comment) 
        cleaned_comments.append(comment.strip())
    return cleaned_comments

def analyze_sentiments(sentiments):
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')

    total_comments = len(sentiments)

    positive_percentage = (positive_count / total_comments) * 100
    negative_percentage = (negative_count / total_comments) * 100

    overall_sentiment = "POSITIVE" if positive_count >= negative_count else "NEGATIVE"

    analysis = {
        "positive_percentage": round(positive_percentage,2),
        "negative_percentage": round(negative_percentage,2),
        "overall_sentiment": overall_sentiment,
        "liked_by_people": "Yes" if overall_sentiment == "POSITIVE" else "No"
    }

    return analysis


def generate_summary(comments):
    prompt = "Summarize the main topic of the following comments in less than 20 words:\n" + "\n".join(comments)
    summary = llm.invoke(prompt)
    return summary.content

def extract_likes(comments):
    prompt = "What do people like about this video based on these comments?Explain under 50 words in pointwise manner \n" + "\n".join(comments)
    likes = llm.invoke(prompt)
    return likes.content

def extract_improvements(comments):
    prompt = "What improvements do people suggest for this video based on these comments?Explain under 50 words in pointwise manner\n" + "\n".join(comments)
    improvements = llm.invoke(prompt)
    return improvements.content

def extract_dislikes(comments):
    prompt = "What do people dislike about this video based on these comments?Explain under 50 words in pointwise manner\n" + "\n".join(comments)
    dislikes = llm.invoke(prompt)
    return dislikes.content



def analyze_comments(comments):
    nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = nlp(comments)
    positive_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment['label'] == 'POSITIVE']
    negative_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment['label'] == 'NEGATIVE']

    summary = generate_summary(comments)
    likes = extract_likes(positive_comments)
    improvements = extract_improvements(negative_comments)
    dislikes = extract_dislikes(negative_comments)
    sentiment_analysis=analyze_sentiments(sentiments)

    analysis = {
         "summary": summary,
        "likes": likes,
        "improvements": improvements,
        "dislikes": dislikes,
        "sentiment_analysis": sentiment_analysis
    }

    print(analysis)

    return analysis


@app.route("/analyze", methods=["POST"])
def analyze():

    json_content = request.json
    video_url = json_content.get("url")

    try:
        comments = get_youtube_comments(video_url)
        if not comments:
            return ({'error': 'No comments found'})

        cleaned_comments = preprocess_comments(comments)
        analysis = analyze_comments(cleaned_comments)
        return (analysis)
    
    except Exception as e:
        return ({'error': str(e)})
    


def start_app():
    app.run(host="0.0.0.0", port=os.getenv("port"), debug=1)
 

if __name__ == "__main__":
    start_app()
