#Instructions to Execute the Code

##Environment Setup:
###1.Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

###2.Install the dependencies
pip install -r requirements.txt

####3.Set NVIDIA API key
On MacOS:
export NVIDIA_API_KEY="your_actual_api_key"
On Windows:
set NVIDIA_API_KEY="your_actual_api_key"

Notes:

1.API Key
The app uses NVIDIA’s API via OpenAI-compatible endpoints. You must get a valid API key from NVIDIA’s developer portal.

2.Model Access
"nvidia/llama-3.2-nv-embedqa-1b-v2" and "nvidia/llama-3.1-nemotron-70b-instruct" models require access privileges.

3.Streamlit
Run the app with:
streamlit run app.py
EXAMPLE: streamlit run c:/Users/ASM9017/Downloads/Nvidia_RAG_Model/Rag_bot_3.py    

4.File Handling
Uploaded PDFs are temporarily saved, so ensure your environment allows temporary file creation.