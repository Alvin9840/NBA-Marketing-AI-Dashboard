"""
Data Analysis Tool
Connects to structured data sources for analysis
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta

class DataTool:
    """Data analysis tool for fan engagement data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_fan_comments(self, game_id: str = None) -> Dict[str, Any]:
        """Load fan comments data"""
        comments_file = self.data_dir / "fan_comments.json"
        if comments_file.exists():
            with open(comments_file, 'r') as f:
                data = json.load(f)
                if game_id:
                    return data if data.get("game_id") == game_id else {}
                return data
        return {}
    
    def load_game_data(self) -> Dict[str, Any]:
        """Load game performance data"""
        game_file = self.data_dir / "game_data.json"
        if game_file.exists():
            with open(game_file, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_sentiment(self, comments_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment from fan comments"""
        comments = comments_data.get("comments", [])
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        platforms = {}
        engagement_total = 0
        
        for comment in comments:
            sentiment = comment.get("sentiment", "neutral")
            sentiment_counts[sentiment] += 1
            
            platform = comment.get("platform", "unknown")
            platforms[platform] = platforms.get(platform, 0) + 1
            
            engagement_total += comment.get("engagement", 0)
        
        total_comments = len(comments)
        sentiment_percentages = {
            sentiment: (count / total_comments * 100) if total_comments > 0 else 0
            for sentiment, count in sentiment_counts.items()
        }
        
        return {
            "total_comments": total_comments,
            "sentiment_distribution": sentiment_percentages,
            "platform_distribution": platforms,
            "average_engagement": engagement_total / total_comments if total_comments > 0 else 0,
            "post_types": comments_data.get("post_types", [])
        }
    
    def get_recent_games(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get games from recent period"""
        game_data = self.load_game_data()
        games = game_data.get("games", [])
        
        recent_games = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for game in games:
            game_date = datetime.strptime(game["date"], "%Y-%m-%d")
            if game_date >= cutoff_date:
                recent_games.append(game)
        
        return recent_games
    
    def analyze_fan_behavior(self) -> Dict[str, Any]:
        """Analyze overall fan behavior patterns"""
        comments_data = self.load_fan_comments()
        sentiment_analysis = self.analyze_sentiment(comments_data)
        
        return {
            "sentiment_analysis": sentiment_analysis,
            "engagement_patterns": {
                "high_engagement_threshold": 1000,
                "viral_potential_comments": len([
                    c for c in comments_data.get("comments", [])
                    if c.get("engagement", 0) > 1000
                ])
            },
            "platform_insights": sentiment_analysis["platform_distribution"]
        }