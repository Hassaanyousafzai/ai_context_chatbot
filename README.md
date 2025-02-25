# Prerequisites

Before running the project, make sure you have the following:

1. **Get Gemini API Key**
   - Visit Google AI Studio and sign up.
   - Generate an API key for Gemini Pro.
   - Store the API key securely.

2. **Get Qdrant API Key**
   - Sign up at Qdrant Cloud.
   - Create a new project.
   - Retrieve your API key from the dashboard.

3. **Get Qdrant URL**
   - Once your Qdrant project is set up, get the Qdrant URL from your instance details.

# Installation

Clone the repository:

```sh
git clone https://github.com/yourusername/resume-roaster.git
cd resume-roaster
```

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

```sh
pip install -r requirements.txt
```

# Environment Variables
Create a .env file in the root directory and add the following:

```sh
GEMINI_API_KEY=your_gemini_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
```

# Usage
Run the application:

```sh
streamlit run frontend.py
```

# Technologies Used
**Python**
- nltk
- qdrant-client
- google-generativeai
- streamlit

**Gemini AI**
**Qdrant (for vector database storage)**
**sentence-transformers**