import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain import prompts
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain

from streamlit_chat import message
import tiktoken
import boto3
import numpy as np
import pandas as pd
import os
import json
import csv
import requests
import datetime
import time
import glob
# from tqdm import tqdm
from googleapiclient.discovery import build
from mtranslate import translate
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi
import configparser
import re
import datetime
# from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

youtube_api_key = ""

### initialize dotenv
load_dotenv(find_dotenv(), override=True)

df_path = "youtube_topic_keyword.csv"
if os.path.isfile(df_path):
    df = pd.read_csv(df_path)
else:
    # df = pd.DataFrame()
    df = pd.DataFrame(
        columns=[
            "youtube_url",
            "video_id",
            "datetime",
            "video_metadata",
            "comment_count",
            "scripts",
            "comment_topics",
            "comment_keywords",
            "comment_list",
        ]
    )

config = configparser.ConfigParser()
config.read("./config.cfg")
## AWS credentials
aws_access_key = config.get("AWS", "aws_access_key")
aws_access_secret_key = config.get("AWS", "aws_access_secret_key")




current_time = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
# utils
@st.cache_resource(max_entries=10, ttl=7200)
def load_document(file_path):
    # pdf load
    file_name_obj, extension = os.path.splitext(file_path)
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)

    return chunks


def print_embedding_cost(texts):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total tokens:{total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens/1000*0.0004:.6f}")


## chroma
def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource(max_entries=10, ttl=7200)
def get_video_comment_all(video_id):
    import googleapiclient.errors

    response_list = []

    next_page_token = None
    while True:
        cnt = len(response_list)
        try:
            response = (
                youtube_client.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=next_page_token if next_page_token else "",
                )
                .execute()
            )
            for item in response.get("items", []):
                response_list.append(item)

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break
        except googleapiclient.errors.HttpError as e:
            print("error")
            break

    comment_threads = sorted(
        response_list,
        key=lambda x: x["snippet"]["topLevelComment"]["snippet"]["likeCount"],
        reverse=True,
    )

    return comment_threads


def visualize_comment(comment_list):

    for idx, comment_obj in enumerate(comment_list):
        comment_snippet = comment_obj["snippet"]["topLevelComment"]["snippet"]
        author = comment_snippet["authorDisplayName"]
        text = comment_snippet["textDisplay"]
        likes = comment_snippet["likeCount"]
        print(f"#{idx+1} Author: {author}")
        print(f"Likes: {likes}")
        print(f"Comment: {text}")
        print()


def visualize_comment_search(comment_list, keyword):

    for idx, comment_obj in enumerate(comment_list):
        comment_snippet = comment_obj["snippet"]["topLevelComment"]["snippet"]
        author = comment_snippet["authorDisplayName"]
        text = comment_snippet["textDisplay"]
        likes = comment_snippet["likeCount"]
        if keyword in text:
            print(f"#{idx+1} Author: {author}")
            print(f"Likes: {likes}")
            print(f"Comment: {text}")
            print()


@st.cache_resource(max_entries=10, ttl=7200)
def save_comment_text(comment_list):

    text_list = []
    for idx, comment_obj in enumerate(comment_list):
        comment_snippet = comment_obj["snippet"]["topLevelComment"]["snippet"]
        author = comment_snippet["authorDisplayName"]
        text = comment_snippet["textDisplay"]
        likes = comment_snippet["likeCount"]
        text_list.append(text)
    return text_list


@st.cache_resource(max_entries=10, ttl=7200)
def trans_to_kor(text_str):
    eng_trans = translate(text_str, "ko")
    return eng_trans


def extract_video_id(url):
    # 일반 YouTube 동영상 URL에서 비디오 ID를 추출하기 위한 정규 표현식
    youtube_pattern = r"(?<=v=)[a-zA-Z0-9_-]{11}"
    # 쇼트 영상 URL에서 비디오 ID를 추출하기 위한 정규 표현식
    shorts_pattern = r"(?<=\/shorts\/)[a-zA-Z0-9_-]+"

    # 입력된 URL이 일반 YouTube 동영상 URL인지 확인
    match_youtube = re.search(youtube_pattern, url)
    # 입력된 URL이 쇼트(Shorts) 영상 URL인지 확인
    match_shorts = re.search(shorts_pattern, url)

    # 일반 YouTube 동영상 URL일 경우 비디오 ID 추출
    if match_youtube:
        return match_youtube.group()
    # 쇼트(Shorts) 영상 URL일 경우 비디오 ID 추출
    elif match_shorts:
        return match_shorts.group()
    # 일반 YouTube 동영상 URL 또는 쇼트(Shorts) 영상 URL이 아닌 경우 None 반환
    else:
        return None


### check comment count
def get_video_comment_count(video_id):
    # youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    # 댓글 통계 정보 요청
    response = youtube_client.videos().list(part="statistics", id=video_id).execute()

    # 댓글 수 반환
    for item in response["items"]:
        comment_count = item["statistics"]["commentCount"]
        return int(comment_count)

### youtube video metdata 
def get_video_metadata(video_id):
    import datetime 
    
    youtube_client = build('youtube','v3',developerKey=youtube_api_key)

    need_keys = ["publishedAt","title","description","thumbnails","channelTitle","tags"]
    request = youtube_client.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()

    video_metadata = response['items'][0]['snippet']
    meta_dict = {key: value for key, value in video_metadata.items() if key in need_keys}
    meta_dict.update({'video_id':video_id})
    meta_dict.update({"save_datetime":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    
    return meta_dict
    
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    return lang


def detect_comments_language(text_list):
    lang_dict = dict()

    for comment_save_obj in text_list:
        lang = detect_language(comment_save_obj)

        if lang not in lang_dict.keys():
            lang_dict[lang] = []
            lang_dict[lang].append(comment_save_obj)
        else:
            lang_dict[lang].append(comment_save_obj)

    return lang_dict


def save_as_json(video_id,row_dict,dir_name='source_data'):

    json_name = f"{video_id}.json"
    json_path = os.path.join(f"{dir_name}", json_name)

    if not os.path.isfile(json_path):
        with open(json_path, "w", encoding="utf-8-sig") as f:
            json.dump(row_dict, f, indent=4, ensure_ascii=False)


### load_json
def load_json_file(json_path):
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8-sig") as json_f:
            json_data = json.load(json_f)
        return json_data
    else:
        print("json_path not Found!")


def print_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page)) for page in texts])
    cost_usd = round(total_tokens/1000 * 0.0004,6)
    
    print(f"Total tokens:{total_tokens}")
    print(f"Embedding Cost in USD:{cost_usd}")
    
    return total_tokens, cost_usd


def print_embedding_cost_chunks(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    cost_usd = round((total_tokens/ 1000) * 0.0004,6)
    print(f"total tokens:{total_tokens}")
    print(f"embedding cost in USD: {total_tokens/ 1000 * 0.0004:.6f}")
    
    return total_tokens, cost_usd


def get_valid_min_token_text_list(text_list):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    max_tokens_limit = 16385 - 2000

    # 각 텍스트의 토큰 수 계산
    text_tokens = [len(enc.encode(text)) for text in text_list]

    selected_texts = []
    total_tokens = 0
    for i, tokens in enumerate(text_tokens):
        if total_tokens + tokens <= max_tokens_limit:
            selected_texts.append(text_list[i])
            total_tokens += tokens
        else:
            break
    print(f"Total tokens: {total_tokens}")

    return selected_texts



## Rendering youtube script and comment summary
def render_youtube_summary(youtube_url):

    ### extract video_id from youtube_url
    video_id = extract_video_id(youtube_url)
    ### checking video_id cache data in source_data as json format
    source_dir = "source_data"
    target_json_name = f"{video_id}.json"
    target_json_path = os.path.join(source_dir, target_json_name)

    json_data = load_json_file(target_json_path)

    ### Rendering Section
    ## 1. video rendering
    # st.video(st.session_state.youtube_url)
    st.video(json_data["youtube_url"])
    ## 2. Content Script
    st.subheader("유튜브 영상 스크립트 요약")
    # st.write(st.session_state.scripts)
    st.write(json_data["scripts"])

    ## 3. Comments Topic
    st.subheader("유튜브 댓글 주요 주제")
    # st.write(st.session_state.topics)
    st.write(json_data["comment_topics"])

    ## 4. Comments Topic keyword
    st.subheader("유튜브 댓글 주제 별 키워드")
    # st.write(st.session_state.keywords)
    st.write(json_data["comment_keywords"])


def show_video_info():
    st.sidebar.header("YouTube Videos")
    target_page = "summary_history"
    
    import glob 
    meta_json_list = glob.glob("meta_dir/*.json", recursive=True)
    # meta_json_list = meta_json_list[:10]
    
    meta_json_dict_sort = []
    for meta_json_path in meta_json_list:
        video_info= load_json_file(meta_json_path)
        
        save_datetime = video_info['save_datetime']
        sort_dict = {
            'meta_json_path':meta_json_path,
            'save_datetime':save_datetime
        }
        meta_json_dict_sort.append(sort_dict)    
    
    meta_json_list_sort = sorted(meta_json_dict_sort, key=lambda x:x['save_datetime'],reverse=True)
    meta_json_list_sort = [meta_json_dict['meta_json_path'] for meta_json_dict in meta_json_list_sort][:10]
    for meta_json_path in meta_json_list_sort:
        video_info = load_json_file(meta_json_path)
        
        title = video_info["title"]
        author = video_info["channelTitle"]
        video_id = video_info["video_id"]
        save_datetime = video_info['save_datetime']
        
        thumbnails_url = video_info['thumbnails']['default']['url']
        thumbnails_path = os.path.join("thumbnails",f"{video_id}.jpg")
        if os.path.isfile(thumbnails_path):
            image_path = thumbnails_path
        else:
            download_and_save_thumbnail(thumbnails_url,video_id)
            image_path = thumbnails_path
            
        target_json_path = os.path.join("source_data", f"{video_id}.json")
        json_data = load_json_file(target_json_path)
        
        # 클릭할 수 있는 링크로 YouTube 제목과 작성자를 출력
        
        
        # if st.sidebar.button(f"{title} - {author}", key=video_id):
        history_side_button = st.sidebar.button(f"{title} \n {author}", key=video_id)
        if history_side_button:
            ##
            st.session_state.page = "summary_history"
            st.write(f"You clicked on {title} by {author}")

            st.session_state.youtube_url = json_data["youtube_url"]
            st.session_state.scripts = json_data["scripts"]
            st.session_state.topics = json_data["comment_topics"]
            st.session_state.keywords = json_data["comment_keywords"]


def render_page():
    if "video_info" in st.session_state:
        show_video_info(st.session_state["video_info"])
    else:
        st.title("Home Page")
        st.write("This is the home page content.")


def reset_session_state():
    if st.button("Reset Session State"):
        st.experimental_rerun()


def load_youtube_shorts(shorts_url):
    # YouTube Shorts 영상을 로드하기 위한 iframe 코드 생성
    html_code = f'<iframe width="560" height="315" src="{shorts_url}" frameborder="0" allowfullscreen></iframe>'
    # 생성된 iframe 코드를 Streamlit에 표시
    st.write(html_code, unsafe_allow_html=True)


## short check
def check_youtube_is_shorts(youtube_url):
    short_string = youtube_url.split("/")[-2]
    if short_string == "shorts":
        return True
    else:
        return False


### save json to s3
def save_json_to_s3(json_path):
    bucket_name = ""
    target_dir = "source_data"
    if os.path.isfile(json_path):
        s3_key = os.path.join(target_dir, os.path.basename(json_path))

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_access_secret_key,
        )

        with open(json_path, "r", encoding="utf-8-sig") as json_f:
            json_data = json.load(json_f)

        ## put json into s3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps(json_data, indent=4, ensure_ascii=False),
        )
    else:
        print(f"Error: File {json_path} does not exist.")
        
def upload_dict_to_s3(target_dir, video_id,input_dict):
    bucket_name = ""
    
    target_json_path = os.path.join(target_dir, f"{video_id}.json")
    ## file as json_data 
    if isinstance(input_dict, dict):
        ## video_id chekcing 
        dict_video_id = input_dict.get('video_id',None)
        assert dict_video_id == video_id
        
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_access_secret_key,
        )
        ## put json into s3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=target_json_path,
            Body=json.dumps(input_dict, indent=4, ensure_ascii=False),
        )
        print(F"UPLOAD SUCCESS TO {bucket_name}/{target_json_path}")
    else:
        print(F"Error: Input dict ERROR")
# def save_log_request(video_id, request_time):
    


def download_and_save_thumbnail(thumbnail_url, video_id):
    import requests
    from PIL import Image
    from io import BytesIO
    file_name = f"{video_id}.jpg"
    file_path = os.path.join("thumbnails", file_name)
    
    response = requests.get(thumbnail_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save(file_path)
        print(f"Thumbnail saved as {file_path}")
    else:
        print("Failed to download thumbnail")

## download mp3 from youtube
@st.cache_resource(max_entries=10, ttl=7200)
def download_mp3_from_youtube(youtube_url, output_dir):
    from pytube import YouTube
    import moviepy.editor as mp

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create a Youtube instance
        youtube = YouTube(youtube_url)
        
        ## Get the highest resolution audio STream 
        audio_stream = youtube.streams.filter(only_audio=True).first()
        if audio_stream:
            audio_file_name = f"{youtube.video_id}.mp4"
            ### audio stream donwload with video_id
            audio_stream.download(output_path=output_dir, filename=audio_file_name)

            ## youtube music download log 
            audio_title = youtube.title
            video_id = youtube.video_id
            duration = youtube.length

            download_file_path = os.path.join(output_dir, audio_file_name)

            ## Load audio clip 
            audio_clip = mp.AudioFileClip(download_file_path)        

            start_time, end_time = 0 , duration

            ## define the start and end time duration 
            subclip = audio_clip.subclip(start_time, end_time)

            ## write subclip 
            target_audio_file_path = os.path.join(output_dir, f"{audio_title}.mp3")
            subclip.write_audiofile(target_audio_file_path)

            ## close audio clip 
            audio_clip.close()
            os.remove(download_file_path)

            print(F"MP3 Download completed : {target_audio_file_path}")
        else:
            print(F"ERRORRRRRRRR")
    except Exception as e:
        print("AN error occurred",e)
        
    return target_audio_file_path

## page
st.set_page_config(page_title="Assistant youtube", page_icon="rocket")

## pages
pages = [
    "simple_chatbot",
    "pdf_agent",
    "유튜브_요약",
    "youtube_comment_sentiment_agent",
    "youtube_music_agent"
]
selected_page = st.sidebar.selectbox("페이지 선택", pages)
## 세션에 저장
st.session_state.page = selected_page

st.sidebar.title("YouTube 링크 히스토리")


show_video_info()
# render_page()



### Instance
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
youtube_client = build("youtube", "v3", developerKey=youtube_api_key)


## session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "scripts" not in st.session_state:
    st.session_state.scripts = []
if "topics" not in st.session_state:
    st.session_state.topics = []
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "eng"  # Default language is English
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = []
if "video_id" not in st.session_state:
    st.session_state.video_id = []
if "submit_button_clicked" not in st.session_state:
    st.session_state.submit_button_clicked = False

### Page select
if st.session_state.page == "simple_chatbot":

    ###
    st.subheader("Chat GPT에게 메세지 요청하는 챗봇")
    system_message = st.text_input(label="Chatbot에게 역할 설정 프롬프트")
    user_prompt = st.text_input(label="Send a message")

    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(SystemMessage(content=system_message))

        # st.write(st.session_state.messages)

    if user_prompt:
        st.session_state.messages.append(HumanMessage(content=user_prompt))

        with st.spinner("Working on your requests...."):
            response = chat(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=response.content))

    # ### render message
    for i, msg in enumerate(st.session_state.messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=f"{i} + ")
        else:
            message(msg.content, is_user=False, key=f"{i}")


elif st.session_state.page == "유튜브_요약":

    ## youtube link
    youtube_url = st.text_input(label="Insert youtube link")
    # st.session_state = dict()
    st.write(st.session_state)

    with st.form(key="youtube_link_submit"):
        submit_button = st.form_submit_button(label="youtube_link_run")
        # text_input = st.text_input(label='입력')
        # st.session_state.text_input = text_input
    if submit_button or st.session_state.submit_button_clicked:

        st.session_state.submit_button_clicked = True
        ## db_row
        # 'youtube_url','video_id','datetime','meta_data','comment_count','comment_topics','comment_keywords','comment_list'
        
        st.session_state.youtube_url = youtube_url
        video_id = extract_video_id(youtube_url)
        st.session_state['video_id'] = video_id
        
        ## Request log data 
        request_ts = datetime.datetime.now().timestamp()
        
        
        ## checking cache, video_id
        source_dir = "source_data"
        target_json_path = os.path.join(source_dir, f"{video_id}.json")
        if os.path.isfile(target_json_path):
            with open(target_json_path, "r", encoding="utf-8-sig") as json_f:
                row_dict = json.load(json_f)
        else:
            row_dict = dict(zip(df.columns.tolist(), [None] * len(df.columns.tolist())))
            # row_dict = dict()
            row_dict["youtube_url"] = youtube_url
            row_dict["video_id"] = video_id
            
            request_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            row_dict['datetime'] = request_time
            ## metadata 
            video_metadata = get_video_metadata(video_id)
            row_dict['video_metadata'] = video_metadata
            
            # st.session_state.youtube_url = youtube_url
            # st.session_state.video_id = video_id

            ### comment_count
            comment_count = (
                row_dict["comment_count"]
                if row_dict["comment_count"]
                else get_video_comment_count(video_id)
            )
            max_count = min(200, comment_count)
            # st.session_state.comment_count = max_count
            # st.write(f'댓글 개수:  {st.session_state.comment_count}')
            row_dict["comment_count"] = comment_count

            ### Extract scripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            script_list = transcript_list.find_transcript(
                ["en", "ko", "es", "ja", "fr"]
            ).fetch()
            script_string = "\n".join([script["text"] for script in script_list])
            ### save script list
            row_dict['script_list'] = script_string

            ## chunks text
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500, chunk_overlap=100
            )
            chunks = text_splitter.create_documents([script_string])
            ## Token, cost_usd checking
            script_token_cnt, script_cost_usd = print_embedding_cost_chunks(chunks)
            
            ### Summurize Scripts using LLM
            map_prompt = """
                Write IN `KOREAN` a short and concise summary of the following:
                Text: `{text}`
                According to summary topics, adding sub-topics and keypoints .
                CONCISE SUMMARY:
            """
            map_prompt_template = PromptTemplate(
                input_variables=["text"], template=map_prompt
            )

            combine_prompt = """
                Write IN `KOREAN` a concise summary of the following text that convers the key points.
                Add a title to the summary. 
                Start your summary with an INTRODUCTION PARAGRAPH that gives an overview. 
                AND CONTENT composed of the main topics as main-bullet-points
                AND sub-topics from specific scripts correspond to main topics as sub-bullet-points.
                AND IF there is problematic subject, provide problematic subject and the answer to that subject.
                AND Below the sub-bullet-points, Add specific sentence with context and example from `Text` as reference to sub-bullet-points.
                if possible AND end the summary with a CONCLUSION PHRASE.
                Text: `{text}`
            """
            combine_prompt_template = PromptTemplate(
                template=combine_prompt, input_variables=["text"]
            )
            ### summary chain map-reduce with map,combine prompt
            summary_chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                map_prompt=map_prompt_template,
                combine_prompt=combine_prompt_template,
                verbose=False,
            )
            script_summary = summary_chain.run(chunks)
            ### Save script summary
            row_dict["scripts"] = script_summary

            ## comment_list
            comment_obj_list = get_video_comment_all(video_id)
            ### preprocess and save comments sorted by likeCount
            comment_save_list = save_comment_text(comment_obj_list)
            row_dict["comment_list"] = comment_save_list

            # comment_strings = "\n ".join(comment_save_list[:max_count])

            valid_comment_list = get_valid_min_token_text_list(comment_save_list)

            comment_strings = "\n ".join(valid_comment_list)

            lang_dict = detect_comments_language(comment_save_list)
            st.write("댓글 언어 개수 현황")
            st.write(f"{key}:{len(val)} \n" for key, val in lang_dict.items())

            ### Extract comment Topic Using LLM Prompt
            if row_dict.get('comment_topics',None) == None:
                topic_template = """
                    이것은 유튜브 댓글들입니다만, 이 유튜브 댓글들에 한하여 무엇을 말하는지 주제를 5가지 정도로 설명 부탁. 
                    만약 5가지 이상으로도 주제를 선정가능하다면 10가지 이내로 부탁.
                    그리고 주제를 이해할 수 있도록 뒷받침하는 근거에 해당하는 소주제도 작성부탁.
                    한국어로.
                    TEXT:{text_list}
                """
                topic_prompt = PromptTemplate(
                    input_variables=["text_list"], template=topic_template
                )
                
                ## topic_prompt token, cost
                topic_strings = topic_template + comment_strings
                topic_token_cnt, topic_cost_usd = print_embedding_cost(topic_strings)
                
                chain = LLMChain(llm=chat, prompt=topic_prompt)
                topic_summary = chain.run({"text_list": comment_strings})

                topic_summary_split = topic_summary.split("\n")

                row_dict["comment_topics"] = topic_summary

            if row_dict.get('comment_keywords', None) == None:
                ### 토픽으로부터 주요 키워드 추출
                keyword_template = """
                    이것은 유튜브 댓글들입니다만, 이 유튜브 댓글들에 한하여 무엇을 말하는지 요약된 주제 리스트 입니다.
                    TOPIC_LIST:{topic_list}
                    이에 해당하는 주제들로부터 상세하게 잘 이해하기 위해서 각 주제 별로 주제에 연관되고 대표하는 키워드들을 제시 부탁드립니다.
                    각 주제 별로 해당하는 관련되고 상세하고 구체적으로 사용된 키워드가 추출되어야 합니다.
                    TOPIC_LIST가 그대로 추출되면 안되고, 상세하게 사용된 키워드를 반환해야됩니다.
                    TEXT:{text_list}
                """

                # keyword_template2 = """
                #     이것은 유튜브 댓글들입니다만, 이 유튜브 댓글들에 한하여 무엇을 말하는지 요약된 주제입니다.
                #     TOPIC_LIST:{topic_list}
                #     이에 해당하는 주제 `TOPIC_LIST`들에 대해 상세하게 잘 이해하기 위해서 각 주제 별로 자주 사용된 단어들을 많이 제시 부탁드립니다.
                #     TEXT:{text_list}
                
                # """
                keyword_prompt = PromptTemplate(
                    input_variables=["topic_list", "text_list"],
                    template=keyword_template,
                    chain_type="map_reduce",
                )
                ## keyword total token, keyword cost_usd 
                keyword_strings = keyword_template + comment_strings + topic_summary
                keyword_token_cnt,keyword_cost_usd = print_embedding_cost(keyword_strings)
                
                keyword_chain = LLMChain(llm=chat, prompt=keyword_prompt)
                keyword_result = keyword_chain.run(
                    {"text_list": comment_strings, "topic_list": topic_summary_split}
                )
                # print(keyword_result)
                # st.session_state.keywords = keyword_result
                row_dict["comment_keywords"] = keyword_result

        ## session_state 설정
        ## youtube_url, video_id , comment_count, scripts, topics
        st.session_state.comment_count = row_dict["comment_count"]
        st.session_state.scripts = row_dict["scripts"]
        st.session_state.script_list = row_dict['script_list']
        st.session_state.comment_list = row_dict["comment_list"]
        st.session_state.topics = row_dict["comment_topics"]
        st.session_state.keywords = row_dict["comment_keywords"]
        st.session_state.metadata = row_dict['video_metadata']
        
        ### SAVE
        ## meta save
        save_as_json(video_id,st.session_state.metadata, dir_name='meta_dir')
        upload_dict_to_s3("meta_dir",video_id, row_dict['video_metadata'])
        
        ### source save
        save_as_json(video_id, row_dict, dir_name='source_data')
        upload_dict_to_s3('source_data', video_id, row_dict)

        
        response_ts = datetime.datetime.now().timestamp()
        diff_ts = (response_ts - request_ts)
        
        log_dict = {
            'youtube_url':youtube_url,
            'video_id':video_id,
            'request_ts':request_ts, 
            'response_ts':response_ts,
            'diff_ts':diff_ts,
            'script_token':script_token_cnt,
            'script_cost_usd':script_cost_usd,
            'topic_token':topic_token_cnt,
            'topic_cost_usd':topic_cost_usd,
            'keyword_token':keyword_token_cnt,
            'keyword_cost_usd':keyword_cost_usd,
            'total_token':script_token_cnt+topic_token_cnt+keyword_token_cnt,
            'total_cost_usd':round(script_cost_usd+topic_cost_usd+keyword_cost_usd,6)
        }
        save_as_json(video_id, log_dict, dir_name='log_dir')
        upload_dict_to_s3('log_dir', video_id, log_dict)
        
        ### Analyze comments language
        lang_dict = detect_comments_language(st.session_state.comment_list)
        # st.write('댓글 언어 개수 현황')
        # st.write(f"{key}:{len(val)} \n" for key, val in lang_dict.items())

        ### Rendering Section

        ## 1. video rendering
        # if check_youtube_is_shorts:
        #     load_youtube_shorts(st.session_state.youtube_url)
        # else:
        st.video(st.session_state.youtube_url)
        ## 2. Content Script
        st.subheader("유튜브 영상 스크립트 요약")
        st.write(st.session_state.scripts)
        # st.markdown(st.session_state.scripts)
        st.subheader("댓글 언어 개수 현황")
        st.write(f"{key}:{len(val)} \n" for key, val in lang_dict.items())
        ## 3. Comments Topic
        st.subheader("유튜브 댓글 주요 주제")
        st.write(st.session_state.topics)
        ## 4. Comments Topic keyword
        st.subheader("유튜브 댓글 주제 별 키워드")
        st.write(st.session_state.keywords)

        ### Translation
        select_lang = st.selectbox(
            "언어선택",
            ["eng", "kor"],
            index=0 if st.session_state.selected_lang == "eng" else 1,
        )
        # Update session state based on selected language
        if select_lang != st.session_state.selected_lang:
            st.session_state.selected_lang = select_lang

        st.write(st.session_state.selected_lang)
        st.subheader("번역으로 ")
        ### Render Topic And Keyword
        if st.session_state.selected_lang == "kor":
            topic_summary = st.session_state.topics
            keyword_summary = st.session_state.keywords
            topic_summary_kor = trans_to_kor(topic_summary)
            keyword_result_kor = trans_to_kor(keyword_summary)
            # st.expander(keyword_result_kor)

            # trans_text = trans_to_kor(text_input)
            # st.write(trans_text)
            st.write(topic_summary_kor)
            st.write(keyword_result_kor)
        else:
            pass

        st.write(st.session_state)


########
## PDF_AGENT
########

elif st.session_state.page == "pdf_agent":

    st.header("PDF_AGENT :")

    st.text_input("Upload PDF file")

    with st.form(key="pdf_file_input"):
        submit_button = st.form_submit_button(label="upload_pdf_file")

    if submit_button or st.session_state.submit_button_clicked:
        pass
        #### pdf_file summarization

        ### Render pdf_file summary

        ### Render Chat Section
        # for i, msg in enumerate(st.session_state.messages[1:]):
        #     if i % 2 == 0:
        #         message(msg.content, is_user=True, key=f"{i} + ")
        #     else:
        #         message(msg.content, is_user=False, key=f"{i}")


### youtube video info store board
elif st.session_state.page == "summary_history":

    ### 세션 내에 있는 것을 적용

    st.write("SUCCESS")
    st.write(st.session_state)
    ## 1. video rendering
    st.video(st.session_state.youtube_url)
    ## 2. Content Script
    st.subheader("유튜브 영상 스크립트 요약")
    st.write(st.session_state.scripts)
    ## 3. Comments Topic
    st.subheader("유튜브 댓글 주요 주제")
    st.write(st.session_state.topics)
    ## 4. Comments Topic keyword
    st.subheader("유튜브 댓글 주제 별 키워드")
    st.write(st.session_state.keywords)


elif st.session_state.page == "session_test":
    st.write(st.session_state)

### Youtube Music Download Page 
elif st.session_state.page == "youtube_music_agent":
    
    ## Page title 
    st.subheader("유튜브 음악을 처리하고 다운로드 하는 페이지 입니다")
    
    ### youtube link url 
    youtube_url = st.text_input(label="Insert youtube link")

    with st.form(key="youtube_link_submit"):
        submit_button = st.form_submit_button(label="youtube_link_run")
    if submit_button or st.session_state.submit_button_clicked:

        st.session_state.submit_button_clicked = True
        
        ## 세션 처리 구현해야 된다 중복처리를 방지하기 위해서 
        
        ## download run 
        output_dir = "audio_dir"
        video_id = extract_video_id(youtube_url)
        origin_music_file_path = os.path.join(output_dir, f"{video_id}.mp4")
        target_audio_file_path = download_mp3_from_youtube(youtube_url, output_dir)

        audio_file_name = os.path.basename(target_audio_file_path)
        st.video(youtube_url)
        
        ## download button 
        with open(target_audio_file_path, 'rb') as f:
                data = f.read()
        st.download_button(label='Click here to download', data=data, file_name=f'{audio_file_name}', mime='audio/mp3')
        
        
