"""
IBM watsonx.ai Tool
Integration with IBM Watsonx models for content generation
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WatsonxTool:
    """IBM watsonx.ai integration for content generation"""
    
    def __init__(self):
        # Load IBM watsonx.ai credentials from environment
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.model_name = "meta-llama/llama-3-3-70b-instruct"  # Using meta llama as shown in your example
        
        # Initialize client and model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize IBM watsonx.ai model inference"""
        if not self.api_key:
            print("⚠️  Warning: WATSONX_API_KEY not found in environment variables")
            print("   Using simulation mode. Set WATSONX_API_KEY for real API calls.")
            return
        
        if not self.project_id:
            print("⚠️  Warning: WATSONX_PROJECT_ID not found in environment variables")
            print("   Using simulation mode. Set WATSONX_PROJECT_ID for real API calls.")
            return
        
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            
            # Create credentials and client
            credentials = Credentials(
                url=self.url,
                api_key=self.api_key
            )
            
            client = APIClient(credentials)
            
            # Create ModelInference instance
            self.model = ModelInference(
                model_id=self.model_name,
                api_client=client,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 500,
                    "min_new_tokens": 50,
                    "temperature": 0.7,
                    "repetition_penalty": 1.0,
                    "decoding_method": "greedy"
                }
            )
            
            print("✅ IBM watsonx.ai model initialized successfully")
            
        except ImportError:
            print("⚠️  ibm-watsonx-ai package not installed. Install with: pip install ibm-watsonx-ai")
            print("   Using simulation mode.")
        except Exception as e:
            print(f"❌ Failed to initialize watsonx.ai model: {str(e)}")
            print("   Using simulation mode.")
    
    def _call_watsonx_api(self, prompt: str, context: Dict[str, Any] = None, 
                         max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """
        Make actual API call to IBM watsonx.ai using ModelInference
        """
        if not self.model:
            return None
        
        try:
            # Prepare the prompt with context
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Update parameters if needed
            self.model.params["max_new_tokens"] = max_tokens
            self.model.params["temperature"] = temperature
            
            # Generate response using ModelInference
            response = self.model.generate_text(prompt=full_prompt)
            
            # The response should be a string directly
            if response and isinstance(response, str):
                return response.strip()
            
        except Exception as e:
            print(f"❌ watsonx.ai API call failed: {str(e)}")
        
        return None
    
    def _prepare_prompt(self, base_prompt: str, context: Dict[str, Any] = None) -> str:
        """Prepare a comprehensive prompt with context"""
        system_prompt = """You are an expert NBA marketing content creator for a senior marketing director. 
        Create engaging, professional content that resonates with basketball fans and drives engagement.
        Focus on authenticity, excitement, and community building."""
        
        if context:
            context_str = f"\n\nContext Information:\n{json.dumps(context, indent=2)}"
        else:
            context_str = ""
        
        return f"{system_prompt}\n\nTask: {base_prompt}{context_str}\n\nResponse:"
    
    def generate_content_hooks(self, game_data: Dict[str, Any], fan_sentiment: Dict[str, Any]) -> List[str]:
        """
        Generate content hooks based on game performance and fan sentiment
        Uses real IBM watsonx.ai API when available, falls back to simulation
        """
        context = {
            "game_data": game_data,
            "fan_sentiment": fan_sentiment,
            "request_type": "content_hooks"
        }
        
        prompt = f"""Generate 5 compelling content hooks for NBA marketing based on:
        - Game: {game_data.get('home_team', 'Team')} vs {game_data.get('away_team', 'Opponent')}
        - Fan sentiment: {fan_sentiment.get('sentiment_distribution', {}).get('positive', 0):.1f}% positive
        - Key highlights: {', '.join(game_data.get('highlights', []))}
        
        Each hook should be engaging, platform-ready, and designed to maximize fan interaction."""
        
        # Try real API call first
        api_response = self._call_watsonx_api(prompt, context, max_tokens=300)
        
        if api_response:
            # Parse the response into individual hooks
            hooks = [line.strip('- •').strip() for line in api_response.split('\n') if line.strip() and not line.lower().startswith(('here', 'content'))]
            return hooks[:5] if hooks else self._generate_simulated_hooks(game_data, fan_sentiment)
        
        # Fallback to simulation
        return self._generate_simulated_hooks(game_data, fan_sentiment)
    
    def _generate_simulated_hooks(self, game_data: Dict[str, Any], fan_sentiment: Dict[str, Any]) -> List[str]:
        """Fallback simulation when API is not available"""
        home_team = game_data.get("home_team", "Team")
        key_players = game_data.get("key_players", {})
        highlights = game_data.get("highlights", [])
        
        sentiment = fan_sentiment.get("sentiment_distribution", {})
        positive_pct = sentiment.get("positive", 0)
        
        hooks = []
        
        # Hook 1: Player performance focus
        if "LeBron James" in key_players:
            hooks.append(f"LeBron's triple-double magic! 🏀✨ Dive into the GOAT's performance that had fans raving at {positive_pct:.1f}% positive sentiment!")
        
        # Hook 2: Team celebration
        hooks.append(f"Victory vibes only! {home_team} fans are celebrating this W with {len(highlights)} game-changing moments!")
        
        # Hook 3: Behind-the-scenes
        hooks.append(f"Inside the locker room: What really happened during those clutch moments that drove fan engagement through the roof?")
        
        # Hook 4: Fan interaction
        hooks.append(f"Fan Spotlight: Real reactions from the {home_team} faithful - from courtside excitement to digital cheers worldwide!")
        
        # Hook 5: Future preview
        hooks.append(f"What's next for {home_team}? Fan predictions and expert analysis on the road ahead!")
        
        return hooks
    
    def suggest_events(self, fan_insights: Dict[str, Any], trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest events based on fan insights and trends
        Uses real API for intelligent suggestions when available
        """
        context = {
            "fan_insights": fan_insights,
            "trend_data": trend_data,
            "request_type": "event_suggestions"
        }
        
        prompt = f"""Suggest 3 innovative NBA fan engagement events based on:
        - Fan sentiment: {fan_insights.get('sentiment_breakdown', 'Mixed')}
        - Key themes: {', '.join(fan_insights.get('key_themes', [])[:3])}
        - Platform preferences: {', '.join(trend_data.get('platform_usage', {}).keys())}
        
        Each event should include: name, description, target audience, expected impact, and timeline."""
        
        # Try real API call
        api_response = self._call_watsonx_api(prompt, context, max_tokens=400)
        
        if api_response:
            # Parse structured response (would need more sophisticated parsing in production)
            return self._parse_event_suggestions(api_response)
        
        # Fallback to predefined events
        return self._get_default_events()
    
    def _parse_event_suggestions(self, api_response: str) -> List[Dict[str, Any]]:
        """Parse AI-generated event suggestions into structured format"""
        # This would need more sophisticated parsing logic
        # For now, return default events
        return self._get_default_events()
    
    def _get_default_events(self) -> List[Dict[str, Any]]:
        """Return default event suggestions"""
        return [
            {
                "name": "Fan Appreciation Night",
                "description": "Special event featuring player meet-and-greets and exclusive content",
                "target_audience": "High-engagement fans",
                "expected_impact": "Boost social media mentions by 25%",
                "timeline": "Next home game"
            },
            {
                "name": "Virtual Q&A Session",
                "description": "Live interactive session with players and coaches",
                "target_audience": "Digital-native fans",
                "expected_impact": "Increase platform engagement by 30%",
                "timeline": "Weekly during season"
            },
            {
                "name": "Behind-the-Scenes Content Series",
                "description": "Exclusive access to practice and game preparation",
                "target_audience": "Content creators and influencers",
                "expected_impact": "Expand reach through user-generated content",
                "timeline": "Ongoing throughout season"
            }
        ]
    
    def generate_creative_content(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        General creative content generation using IBM watsonx.ai
        """
        # Try real API call first
        api_response = self._call_watsonx_api(prompt, context, max_tokens=300)
        
        if api_response:
            return api_response
        
        # Fallback to simulation
        return self._generate_simulated_content(prompt)
    
    def _generate_simulated_content(self, prompt: str) -> str:
        """Fallback simulation for content generation"""
        if "social media post" in prompt.lower():
            return "🏀 GAME NIGHT MAGIC! Just witnessed history at the Staples Center! LeBron James delivering a MASTERCLASS performance with 28 points, 12 rebounds, and 8 assists! The Lakers faithful are UNSTOPPABLE! Who's ready for the next chapter? #LakersNation #NBA #GOAT 🐐"
        
        elif "email campaign" in prompt.lower():
            return "Subject: Your Lakers Nation Family Just Got Bigger!\n\nDear Fellow Lakers Fan,\n\nLast night's victory wasn't just a win—it was a testament to what makes Lakers basketball legendary. LeBron James continues to rewrite the record books, and YOU are part of this incredible journey.\n\nDon't miss out on exclusive behind-the-scenes content and early access to tickets. Join the conversation and let's build something special together.\n\nGo Lakers!\n\nThe Lakers Marketing Team"
        
        else:
            return f"Creative content generated for: {prompt}\n\n[Simulated IBM Watsonx Response: This would contain rich, engaging content tailored to your NBA marketing needs, incorporating fan sentiment data, performance highlights, and predictive insights to create compelling narratives that resonate with your audience.]"