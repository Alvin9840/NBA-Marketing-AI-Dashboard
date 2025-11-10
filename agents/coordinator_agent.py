"""
Coordinator Agent - The Manager
Uses BeeAI Workflow to coordinate specialist agents
"""

import json
from typing import Dict, List, Any
from pathlib import Path

# Import specialist agents
from .sentiment_agent import SentimentAgent
from .creative_agent import CreativeAgent
from .predictive_agent import PredictiveAgent

class CoordinatorAgent:
    """
    The Coordinator Agent acts as the "brain" of the operation.
    It receives director prompts, reasons about them, and delegates
    tasks to specialist agents using BeeAI Workflow patterns.
    """
    
    def __init__(self):
        self.sentiment_agent = SentimentAgent()
        self.creative_agent = CreativeAgent()
        self.predictive_agent = PredictiveAgent()
        self.workflow_state = {}
    
    def process_director_request(self, request: str) -> Dict[str, Any]:
        """
        Main entry point for director requests.
        Analyzes the request and orchestrates specialist agents.
        """
        print(f"🤖 Coordinator Agent: Processing request: '{request}'")
        
        # Analyze request to determine which agents to involve
        analysis = self._analyze_request(request)
        
        # Initialize workflow
        workflow_result = {
            "request": request,
            "analysis": analysis,
            "tasks": [],
            "results": {},
            "final_response": ""
        }
        
        # Execute workflow based on analysis
        if analysis["requires_sentiment"]:
            print("📊 Delegating to Sentiment Agent...")
            sentiment_result = self.sentiment_agent.analyze_sentiment(
                analysis["sentiment_params"]
            )
            workflow_result["results"]["sentiment"] = sentiment_result
            workflow_result["tasks"].append("sentiment_analysis")
        
        if analysis["requires_content"]:
            print("🎨 Delegating to Creative Agent...")
            content_result = self.creative_agent.generate_content(
                analysis["content_params"]
            )
            workflow_result["results"]["content"] = content_result
            workflow_result["tasks"].append("content_generation")
        
        if analysis["requires_prediction"]:
            print("🔮 Delegating to Predictive Agent...")
            prediction_result = self.predictive_agent.forecast_trends(
                analysis["prediction_params"]
            )
            workflow_result["results"]["prediction"] = prediction_result
            workflow_result["tasks"].append("trend_forecasting")
        
        # Synthesize final response
        final_response = self._synthesize_response(workflow_result)
        workflow_result["final_response"] = final_response
        
        print("✅ Coordinator Agent: Workflow completed")
        return workflow_result
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """
        Analyze the director's request to determine required agents and parameters
        """
        request_lower = request.lower()
        
        analysis = {
            "requires_sentiment": False,
            "requires_content": False,
            "requires_prediction": False,
            "sentiment_params": {},
            "content_params": {},
            "prediction_params": {}
        }
        
        # Sentiment analysis triggers
        sentiment_keywords = [
            "sentiment", "fans saying", "fan reaction", "comments",
            "posts", "engagement", "analyze fans"
        ]
        if any(keyword in request_lower for keyword in sentiment_keywords):
            analysis["requires_sentiment"] = True
            analysis["sentiment_params"] = {
                "focus": "recent_game",
                "depth": "detailed"
            }
        
        # Content generation triggers
        content_keywords = [
            "content hooks", "social media", "posts", "content",
            "marketing materials", "campaign", "suggest events"
        ]
        if any(keyword in request_lower for keyword in content_keywords):
            analysis["requires_content"] = True
            analysis["content_params"] = {
                "type": "hooks_and_events",
                "context": "recent_performance"
            }
        
        # Prediction triggers
        prediction_keywords = [
            "forecast", "future", "trend", "predict", "next",
            "what should we do", "planning"
        ]
        if any(keyword in request_lower for keyword in prediction_keywords):
            analysis["requires_prediction"] = True
            analysis["prediction_params"] = {
                "timeframe": "next_month",
                "scope": "comprehensive"
            }
        
        return analysis
    
    def _synthesize_response(self, workflow_result: Dict[str, Any]) -> str:
        """
        Synthesize a comprehensive response from all agent results
        """
        results = workflow_result["results"]
        
        response_parts = []
        response_parts.append("🎯 **NBA Fan Engagement Analysis & Recommendations**\n")
        
        # Sentiment section
        if "sentiment" in results:
            sentiment = results["sentiment"]
            response_parts.append("📊 **Fan Sentiment Analysis:**")
            response_parts.append(f"• Total comments analyzed: {sentiment['total_comments']}")
            response_parts.append(f"• Sentiment breakdown: {sentiment['sentiment_breakdown']}")
            response_parts.append(f"• Key themes: {', '.join(sentiment['key_themes'][:3])}")
            response_parts.append("")
        
        # Content section
        if "content" in results:
            content = results["content"]
            response_parts.append("🎨 **Content Recommendations:**")
            for hook in content["content_hooks"][:3]:
                response_parts.append(f"• {hook}")
            response_parts.append("")
            
            if content["event_suggestions"]:
                response_parts.append("📅 **Suggested Events:**")
                for event in content["event_suggestions"][:2]:
                    response_parts.append(f"• **{event['name']}**: {event['description']}")
                    response_parts.append(f"  Expected impact: {event['expected_impact']}")
                response_parts.append("")
        
        # Prediction section
        if "prediction" in results:
            prediction = results["prediction"]
            response_parts.append("🔮 **Trend Forecast & Strategic Recommendations:**")
            response_parts.append(f"• Next month engagement prediction: {prediction['engagement_forecast']}")
            response_parts.append(f"• Key opportunities: {', '.join(prediction['opportunities'][:3])}")
            response_parts.append("")
        
        # Executive summary
        response_parts.append("💡 **Executive Summary:**")
        response_parts.append("Your fan engagement strategy is positioned for growth. Focus on interactive content,")
        response_parts.append("leverage positive sentiment momentum, and prepare for emerging digital trends.")
        response_parts.append("The AI-driven insights above provide a reliable foundation for your planning.")
        
        return "\n".join(response_parts)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status for monitoring"""
        return {
            "active_tasks": len(self.workflow_state),
            "completed_workflows": 0,  # Would track in production
            "agent_status": {
                "sentiment_agent": "ready",
                "creative_agent": "ready", 
                "predictive_agent": "ready"
            }
        }