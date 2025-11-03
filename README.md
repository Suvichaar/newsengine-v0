# Web Story Content Generator

A Streamlit application that automatically generates complete AMP Web Stories from news article URLs, including text-to-speech audio generation and HTML output.

## Features

- 🔄 **One-Click Generation**: Just provide an article URL and number of slides, get a complete web story
- 🎙️ **Text-to-Speech**: Auto-generates audio using Azure Neural Voices
- 📝 **Content AI**: GPT-powered content extraction and summarization
- 🎨 **AMP Web Stories**: Automatically generates compliant HTML
- ☁️ **S3 Integration**: Uploads audio files directly to AWS S3
- 🔍 **SEO Optimized**: Auto-generates meta tags, descriptions, and keywords

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Copy the example secrets file and fill in your credentials:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your API keys:

```toml
[azure_api]
AZURE_OPENAI_ENDPOINT = "your-azure-openai-endpoint"
AZURE_OPENAI_API_KEY = "your-azure-openai-api-key"

[azure]
AZURE_API_KEY = "your-azure-speech-api-key"
AZURE_REGION = "eastus"

[aws]
AWS_ACCESS_KEY = "your-aws-access-key"
AWS_SECRET_KEY = "your-aws-secret-key"
AWS_REGION = "ap-south-1"
AWS_BUCKET = "your-bucket-name"
S3_PREFIX = "media/"
CDN_BASE = "https://your-cdn-url/"
```

### 3. Run the Application

```bash
streamlit run app-new.py
```

## Usage

1. Enter a news article URL
2. Specify the number of slides (8-10 total, including title and hook)
3. Click "Generate Complete Web Story"
4. Download the generated HTML file

## Output

The application generates:
- **AMP Web Story HTML**: Fully compliant web story ready for hosting
- **Audio files**: Text-to-speech audio for each slide uploaded to S3
- **SEO metadata**: Auto-generated meta tags and descriptions

## Requirements

- Python 3.8+
- Azure OpenAI API key
- Azure Speech Services API key
- AWS S3 credentials
- Internet connection for article extraction

## License

MIT

