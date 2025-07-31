# research-assit-itai-2376
One student project for AI research assistant
This project implements a Virtual Research Assistant for the ITAI 2376 Deep Learning Final Project, inspired by ojasskapre's research assistant and LangChain tutorials. The agent gathers sources, summarizes key points, identifies connections, and formats citations using DuckDuckGo Search and Wikipedia, with a feedback system for reinforcement learning.

## Setup
1. **Google Drive**: Mount your drive in Colab (Cell 2).
2. **Folder Path**: Set the `folder` variable in Cell 5 to your Google Drive folder (e.g., `/content/drive/My Drive/`).
3. **API Key**: Add your OpenAI API key to `.env` in the same folder (Cell 6).
4. **Dependencies**: Install packages in Cell 4.

## Usage
- **Run Cells**: Execute Cells 1–12 in order.
- **UI (Cell 12)**:
  - Enter a query (e.g., “what is langchain”) in the text box.
  - Click “Submit Question” to generate a report.
  - Adjust the `IntSlider` (1–5) for feedback and click “Submit Feedback”.
  - Reports are saved as `research_summary.pdf` in the `folder` path (e.g., `/content/drive/My Drive/Colab Notebooks/research assistant final/research_summary.pdf`).
- **Output**: Check token usage (`Tokens used: ...`) and PDF in Google Drive.

## Model and Code
- **Model**: Uses `GPF-3.5 turbo`, maybe `GPT-4.1 nano`later ($0.10/1M input, $0.40/1M output, `max_tokens=200`) for cost-effective query processing.
- **Core Functions** (Cell 11):
  - `process_request`: Handles queries with caching and 429 retries (uses `time`).
  - `record_feedback`: Stores feedback (1–5 scale) via `FeedbackSystem`.
  - `search_and_summarize`: Generates reports, tracks tokens with `tiktoken`.

## Troubleshooting
- **429 Quota Error**: Ensure your OpenAI account has credits (Cell 6). Apply for OpenAI’s Researcher Access Program for subsidized credits.
- **PDF Errors**: Install `weasyprint` dependencies (Cell 4: `!apt-get install -y libpango-1.0-0 libpangoft2-1.0-0`).

## References
- [ojasskapre/langchain-apps](https://github.com/ojasskapre/langchain-apps)
- [LangChain Tutorial Video](https://youtu.be/DjuXACWYkkU?si=_v1Yz0R9ygpTopP3)
