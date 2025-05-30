"""
LLM Configuration for Reddit Mention Tracker
"""
import os
from typing import Optional

class LLMConfig:
    """Configuration class for LLM integration."""
    
    def __init__(self):
        self.groq_api_key = 'GROQ KEY'  # Replace with your actual valid Groq API key
        self.model_name = "deepseek-r1-distill-llama-70b"
        self.max_tokens = 4096
        self.temperature = 0.1
        self.max_context_length = 60000  # Conservative estimate for context window
        self.max_mentions_per_analysis = 50  # Limit mentions to prevent context overflow
    
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return self.groq_api_key is not None and len(self.groq_api_key.strip()) > 0
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for LLM configuration."""
        return """
## 🤖 AI Analysis Setup Instructions

To enable AI-powered analysis of your Reddit data, you need to configure a Groq API key:

### Step 1: Get a Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key

### Step 2: Set Environment Variable
#### On Windows:
```bash
set GROQ_API_KEY=your_api_key_here
```

#### On Linux/Mac:
```bash
export GROQ_API_KEY=your_api_key_here
```

#### Or create a .env file:
```
GROQ_API_KEY=your_api_key_here
```

### Step 3: Restart the Application
After setting the API key, restart the Reddit Mention Tracker for changes to take effect.

### Features Available with AI Analysis:
- **Sentiment Analysis**: Deep insights into public opinion
- **Theme Identification**: Key topics and discussion patterns
- **Trend Detection**: Emerging patterns and viral content
- **Community Analysis**: Subreddit-specific insights
- **Competitive Intelligence**: Market positioning analysis
- **Content Recommendations**: Actionable insights for improvement

### Model Information:
- **Model**: Groq's deepseek-r1-distill-llama-70b
- **Strengths**: Fast inference, analytical reasoning, data interpretation
- **Context Window**: Large context for comprehensive analysis
- **Cost**: Groq provides competitive pricing for API usage
        """

# Global configuration instance
llm_config = LLMConfig() 