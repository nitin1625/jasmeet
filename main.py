import streamlit as st
from dotenv import load_dotenv
import os
from back import get_youtube_comments, preprocess_comments, analyze_comments


load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')


def analyze(video_url):
    try:
        comments = get_youtube_comments(video_url, youtube_api_key)
        if not comments:
            return ({'error': 'No comments found'})

        cleaned_comments = preprocess_comments(comments)
        analysis = analyze_comments(cleaned_comments)
        return (analysis)
    except Exception as e:
        return ({'error': str(e)})
    

st.title("YouTube Comment Analyzer")
video_url = st.text_input("Enter a YouTube video URL:")


if st.button("Analyze Comments"):
    if not video_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        with st.spinner("Analyzing..."):
            try:
                    analysis = analyze(video_url)
                    st.json(analysis)
            except:
                st.error(analysis)


