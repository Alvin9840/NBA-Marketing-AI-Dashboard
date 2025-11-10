"""
Creative Agent - Content Generation Specialist
Uses IBM watsonx.ai for creative content and event planning
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from tools.watsonx_tool import WatsonxTool
from tools.data_tool import DataTool

class CreativeAgent:
    """
    Specialist agent for creative content generation.
    Generates new ideas and content using IBM watsonx.ai.
    """
    
    def __init__(self):
        self.watsonx_tool = WatsonxTool()
        self.data_tool = DataTool()
    
    def generate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main content generation workflow
        """
        print("🎨 Creative Agent: Starting content generation...")
        
        # Load recent game data for context
        game_data = self.data_tool.load_game_data()
        recent_games = game_data.get("games", [])[:1]  # Most recent game
        
        # Get fan sentiment for content tailoring
        fan_comments = self.data_tool.load_fan_comments()
        sentiment_analysis = self.data_tool.analyze_sentiment(fan_comments)
        
        # Generate content based on parameters
        result = {}
        
        if params.get("type") == "hooks_and_events":
            # Generate content hooks
            if recent_games:
                result["content_hooks"] = self.watsonx_tool.generate_content_hooks(
                    recent_games[0], sentiment_analysis
                )
            
            # Generate event suggestions
            result["event_suggestions"] = self.watsonx_tool.suggest_events(
                sentiment_analysis, {}
            )
        
        # Generate additional creative content
        result["sample_content"] = self._generate_sample_content(params)
        
        print("✅ Creative Agent: Content generation completed")
        return result
    
    def _generate_sample_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample content pieces"""
        return {
            "social_media_post": self.watsonx_tool.generate_creative_content(
                "Create an engaging social media post about the recent Lakers victory"
            ),
            "email_campaign": self.watsonx_tool.generate_creative_content(
                "Draft an email campaign for fan engagement"
            ),
            "content_strategy": {
                "primary_focus": "Interactive fan experiences",
                "secondary_focus": "Player storytelling",
                "tertiary_focus": "Community building"
            }
        }
    
    def generate_event_plan(self, event_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed event planning"""
        base_events = self.watsonx_tool.suggest_events({}, {})
        
        # Enhance with specific planning details
        for event in base_events:
            if event["name"].lower().replace(" ", "_") == event_type.lower().replace(" ", "_"):
                event["detailed_plan"] = {
                    "objectives": [
                        "Increase fan engagement by 25%",
                        "Boost social media mentions",
                        "Create memorable experiences"
                    ],
                    "target_metrics": {
                        "attendance": "500+ fans",
                        "social_engagement": "10K+ interactions",
                        "sentiment_lift": "+15% positive sentiment"
                    },
                    "execution_steps": [
                        "Pre-event promotion (1 week)",
                        "On-site coordination",
                        "Live social media coverage",
                        "Post-event engagement"
                    ],
                    "budget_estimate": "$15,000 - $25,000",
                    "timeline": event["timeline"]
                }
                return event
        
        return base_events[0] if base_events else {}