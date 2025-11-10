# NBA Fan Engagement AI Tool

An agentic AI system designed to revolutionize fan engagement for NBA marketing directors. This tool provides automated fan sentiment analysis, AI-driven event planning, and predictive trend forecasting to make planning easy and reliable.

## 🎯 Target User
**Senior NBA Marketing Director** - Experience "relief" and "easy, reliable planning" through AI-powered insights.

## 🏗️ Architecture

### The Coordinator Agent (The Manager)
- **Framework**: BeeAI Workflow
- **Role**: Analyzes director requests and delegates to specialist agents
- **Communication**: Uses Agent Communication Protocol (ACP)

### Specialist Agents (The Team)

1. **SentimentAgent** - Fan Sentiment Analysis
   - Uses RAG and Data Analysis tools
   - Analyzes fan posts and comments
   - Provides insights on fan needs

2. **CreativeAgent** - Content Generation
   - Uses IBM watsonx.ai (Granite models)
   - Generates content hooks and event suggestions
   - Creates compelling marketing materials

3. **PredictiveAgent** - Trend Forecasting
   - Uses predictive analytics
   - Forecasts fan behavior and trends
   - Provides strategic recommendations

## 🚀 Features

### Automated Fan Sentiment Analysis
- Understand fan needs through post analysis
- Real-time sentiment monitoring
- Platform-specific engagement insights

### AI-Driven Event & Content Planning
- Content hooks based on recent performances
- Event planning recommendations
- Automated creative content generation

### Predictive Trend Forecasting
- Future fan behavior analysis
- Competitive advantage insights
- Strategic planning support

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
  
   pip install -r requirements.txt
   3. Set up your IBM watsonx.ai credentials (for production)

4. Run the application:
  
   python main.py
   ## 🎮 Usage

### Interactive Mode
python main.py### Demo Mode
python main.py demo### Example Queries
- "Summarize what fans are saying about our last game and suggest content hooks"
- "What are fans saying about the Lakers vs Warriors game?"
- "Generate content hooks based on recent performances"
- "Analyze future trends and forecast fan behavior"

## 📊 Sample Data

The tool includes mock data for:
- Fan comments from various platforms (Twitter, Reddit, Instagram, etc.)
- Game performance data
- Historical engagement trends

## 🔧 Configuration

- `config/beeai_config.yaml`: BeeAI workflow configuration
- `config/saiber_tools.yaml`: Tool endpoints and parameters
- `data/`: Sample data files
- `knowledge/`: Knowledge base files (mock)

## 🤖 AI Foundation

Built on **BeeAI Framework** and **IBM watsonx.ai**:
- **Backend Brain**: IBM Granite models for reasoning and generation
- **Agent Communication**: Structured protocol for agent coordination
- **Tool Integration**: Seamless connection to data sources and AI models

## 📈 Benefits

✅ **Relief**: Automated insights reduce manual research burden  
✅ **Easy Planning**: AI-generated recommendations simplify decision-making  
✅ **Reliable**: Data-driven predictions increase confidence  
✅ **Competitive Advantage**: Stay ahead with predictive analytics  
✅ **Scalable**: Handles multiple games and fan interactions simultaneously  

## 🔒 Privacy & Ethics

- All fan data handling follows NBA privacy guidelines
- Sentiment analysis respects user consent
- Predictive models use aggregated, anonymized data
- Content generation avoids harmful or biased outputs