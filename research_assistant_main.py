# -*- coding: utf-8 -*-
"""research assistant main.ipynb


# ITAI 2376 Research Assistant Agent

This notebook implements a Virtual Research Assistant for the ITAI 2376 Deep Learning Final Project. Inspired by ojasskapre's research assistant project and langchain tutorial video, this agent gathers sources, summarizes key points, identifies connections, and formats citations. It uses DuckDuckGo Search and Wikipedia as tools, includes reinforcement learning via a feedback system, and follows safety measures\
Reference: \
https://github.com/ojasskapre/langchain-apps/blob/main/Readme.md \
https://youtu.be/DjuXACWYkkU?si=_v1Yz0R9ygpTopP3

##1. Environment set up
Mount Google Drive
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""2. Install pip packages"""

import os

# Install system dependencies for weasyprint
# Install Python packages
!apt-get update -q
!apt-get install -y -q libpango-1.0-0 libpangoft2-1.0-0
!pip install -q openai langchain langchain-openai langchain-community tiktoken ddgs wikipedia wikipedia-api python-dotenv numpy ipywidgets weasyprint

"""5.Inspect all necessary files and Load `.env.example` manually"""

from dotenv import load_dotenv

# Define the path to your .env file in Google Drive
# You'll need to adjust this path to match your file's location
folder = '/your folder path'
files = os.listdir(folder)
print("Files in folder:\n", files)
#os.path.exists(dotenv_path)

"""#6. load environment variables
.env loading with API ke
"""

dotenv_path = '/your folder path/.env'
# Load environment variables from the .env file
load_dotenv(dotenv_path)

# Access the API key using os.getenv()
# Replace 'YOUR_API_KEY_NAME' with the actual variable name in your .env file
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not found in .env file.')
if len(api_key) < 20:  # Basic validation
    raise ValueError(f'Invalid OPENAI_API_KEY: {api_key[:8]}')
# Replace 'wei-itai' with your actual OpenAI API key in your .env file
print('API Key loaded successfully (first few characters):', api_key[:8] + '...')

"""# 7. Prompt Templates and Import prompt functions"""

from langchain.prompts import PromptTemplate

def get_search_prompt():
    return PromptTemplate(
        input_variables=["question"],
        template="Generate 5 effective Google search queries to answer the following question:\n\n{question}"
    )

def get_web_prompt():
    return PromptTemplate(
        input_variables=["question", "summaries"],
        template=(
            "Use the following extracted web page summaries to answer the question.\n"
            "If the information is insufficient, reply with 'Not enough information.'\n\n"
            "Question: {question}\n\nSummaries:\n{summaries}"
        )
    )

def get_report_prompt():
    return PromptTemplate(
        input_variables=["question", "answer"],
        template="Write a professional, concise research summary answering the question: {question}\n\nAnswer:\n{answer}"
    )

"""8. Import LangChain modules and BeautifulSoup imports"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from bs4 import BeautifulSoup

"""## 9. load utility functions and and toolkit (DuckDuckGo, Wikipedia)"""

import weasyprint
import io
from bs4 import BeautifulSoup
import requests
from ddgs import DDGS
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import time

# Constants
RESULTS_PER_QUESTION = 3  # Number of web search results per query

# Initialize Wikipedia tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def download_as_pdf(content):
    try:
        html = weasyprint.HTML(string=content)
        pdf = io.BytesIO()
        html.write_pdf(pdf)
        return pdf.getvalue()
    except Exception as e:
        print(f"PDF conversion error: {e}")
        return None

def scrape_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text if text else "No text found."
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

def web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r['href'] for r in ddgs.text(query, max_results=RESULTS_PER_QUESTION)]
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def safe_tool_call(tool_function, *args, **kwargs):
    try:
        start_time = time.time()
        result = tool_function(*args, **kwargs)
        if time.time() - start_time > 5:
            return {'status': 'error', 'message': 'Tool call took too long.'}
        return {'status': 'success', 'data': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

"""##10 Feedback System"""

from datetime import datetime

class FeedbackSystem:
    def __init__(self):
        self._feedback_history = []
        self._policy_weights = {'relevance': 0.5, 'helpfulness': 0.3, 'safety': 0.2}

    def record_feedback(self, response_id, feedback_type, value):
        """Record feedback for a specific type."""
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError("Feedback value must be an integer between 1 and 5.")
        if feedback_type not in ['relevance', 'helpfulness', 'safety']:
            raise ValueError("Invalid feedback type.")
        self._feedback_history.append({
            'response_id': response_id,
            'type': feedback_type,
            'value': value,
            'timestamp': datetime.now()
        })
        self._update_policy()

    def _update_policy(self):
        """Update policy based on recent feedback."""
        recent_feedback = self._feedback_history[-10:]
        if recent_feedback:
            for ftype in ['relevance', 'helpfulness', 'safety']:
                avg = sum(f['value'] for f in recent_feedback if f['type'] == ftype) / len([f for f in recent_feedback if f['type'] == ftype]) if any(f['type'] == ftype for f in recent_feedback) else 3
                if avg < 3.5:
                    self._policy_weights[ftype] += 0.05
            sum_weights = sum(self._policy_weights.values())
            for key in self._policy_weights:
                self._policy_weights[key] /= sum_weights
    def get_policy_weights(self):
        return self._policy_weights

"""Agent Architecture\
add rate limit handling and token estimation modules

11. User Interface with Colab Forms
"""

import time
import random
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import uuid

class ResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=api_key, max_tokens=200)
        self.feedback_system = FeedbackSystem()
        self.memory = []  # Cache for queries
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Explicit tokenizer

    def validate_input(self, question):
        """Validate user input for safety and format."""
        if not isinstance(question, str):
            return False, "Question must be a string."
        if len(question) < 5:
            return False, "Question too short (min 5 characters)."
        if any(term in question.lower() for term in ["harm", "illegal"]):
            return False, "Question contains inappropriate content."
        return True, None

    def process_request(self, question):
        is_valid, error = self.validate_input(question)
        if not is_valid:
            return {'status': 'error', 'message': error}

        # Check cache
        for item in self.memory:
            if item['question'].lower() == question.lower():
                # Check most recent feedback score
                response_id = item['response_id']
                feedback = self.feedback_system._feedback_history
                recent_score = next((f['value'] for f in reversed(feedback) if f['response_id'] == response_id), 5)
                if recent_score >= 4:
                    return {'status': 'success', 'report': item['report'], 'response_id': response_id}
                else:
                    print(f"Low feedback score ({recent_score}) for {response_id}. Refreshing response.")

        retries = 3
        for attempt in range(retries):
            try:
                report, response_id = self.search_and_summarize(question)
                self.memory.append({'question': question, 'report': report, 'response_id': response_id})
                return {'status': 'success', 'report': report, 'response_id': response_id}
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                    continue
                return {'status': 'error', 'message': f'Failed to process request: {e}'}

    def record_feedback(self, response_id, relevance, helpfulness, safety):
        try:
            self.feedback_system.record_feedback(response_id, 'relevance', relevance)
            self.feedback_system.record_feedback(response_id, 'helpfulness', helpfulness)
            self.feedback_system.record_feedback(response_id, 'safety', safety)
            return True
        except Exception as e:
            print(f"Feedback error: {e}")
            return False

    def search_and_summarize(self, question):
        prompt_text = get_search_prompt().format(question=question)
        tokens = len(self.encoding.encode(prompt_text))
        print(f"Tokens used: {tokens}")
        search_chain = RunnableSequence(self.llm | StrOutputParser())
        queries = search_chain.invoke(prompt_text).split('\n')

        # Fetch web and Wiki content
        summaries = []
        citations = []
        for query in queries:
            web_urls = web_search(query)
            for url in web_urls:
                text = scrape_text(url)
                if text and not text.startswith('Failed'):
                    summaries.append(text[:500])
                    citations.append(url)
            wiki_result = safe_tool_call(wikipedia_tool.run, query)  # Uses wikipedia_tool from Cell 9
            if wiki_result['status'] == 'success':
                summaries.append(wiki_result['data'][:500])
                citations.append(f"Wikipedia: {query}")

        if not summaries:
            summaries = ["No relevant content found."]

        web_prompt = get_web_prompt().format(question=question, summaries="\n".join(summaries))
        summary = self.llm.invoke(web_prompt).content
        report_prompt = get_report_prompt().format(question=question, answer=summary)
        report_body = self.llm.invoke(report_prompt).content
        # Format report with title and citations
        report = f"# {question.title()}\n\n{report_body}\n\n## References\n"
        for i, citation in enumerate(citations, 1):
            report += f"{i}. {citation}\n"
        response_id = str(uuid.uuid4())
        return report, response_id

"""12.User Interface with IntSlider and PDF output"""

import os
import ipywidgets as widgets
from IPython.display import display, Markdown

# Initialize agent
agent = ResearchAssistant()  # Instantiate ResearchAssistant

# Initialize response_id for feedback
response_id = None

# Create UI elements
question_input = widgets.Text(
    value='',
    placeholder='Enter your research question here',
    description='',
    layout={'width': '500px'}
)

submit_button = widgets.Button(
    description='Submit Question',
    button_style='success',
    tooltip='Click to submit your question'
)

feedback_value = widgets.IntSlider(
    value=3,
    min=1,
    max=5,
    step=1,
    description='Feedback Score:',
    layout={'width': '500px'}
)

feedback_button = widgets.Button(
    description='Submit Feedback',
    button_style='info',
    tooltip='Click to submit feedback'
)

output = widgets.Output()

# Callback for submit button
def on_submit_clicked(b):
    global response_id
    with output:
        output.clear_output()
        question = question_input.value
        result = agent.process_request(question)
        if result['status'] == 'success':
            display(Markdown('### Research summary\n' + result['report']))
            response_id = result['response_id']
            # Save report as PDF
            pdf_data = download_as_pdf(result['report'])
            if pdf_data:
                pdf_path = os.path.join(folder, 'research_summary.pdf')
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_data)
                print(f'Report saved as {pdf_path}')
        else:
            print('Error:', result['message'])

# Callback for feedback button
def on_feedback_clicked(b):
    with output:
        output.clear_output()
        if response_id:
            success = agent.record_feedback(response_id, feedback_value.value, feedback_value.value, feedback_value.value)
            if success:
                print(f'Feedback recorded for response ID: {response_id}')
            else:
                print('Error recording feedback')
        else:
            print('No response to provide feedback for')

# Connect callbacks to buttons
submit_button.on_click(on_submit_clicked)
feedback_button.on_click(on_feedback_clicked)

# Display UI
display(question_input, submit_button, feedback_value, feedback_button, output)