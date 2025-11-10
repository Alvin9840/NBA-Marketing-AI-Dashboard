"""
Sentiment Agent - Fan Sentiment Analysis Specialist
Uses RAG and Data Analysis for understanding fan needs
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from tools.rag_tool import RAGTool
from tools.data_tool import DataTool

class SentimentAgent:
    """
    Specialist agent for fan sentiment analysis.
    Handles analyzing past and present data to understand fan needs.
    """
    
    def __init__(self):
        self.rag_tool = RAGTool()
        self.data_tool = DataTool()
    
    def analyze_sentiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main sentiment analysis workflow
        """
        print("🔍 Sentiment Agent: Starting fan sentiment analysis...")
        
        # Load current fan data
        fan_comments = self.data_tool.load_fan_comments()
        sentiment_analysis = self.data_tool.analyze_sentiment(fan_comments)
        
        # Get historical insights from knowledge base
        historical_insights = self.rag_tool.retrieve_fan_insights("sentiment analysis")
        
        # Combine and analyze
        result = self._synthesize_sentiment_analysis(
            sentiment_analysis, 
            historical_insights,
            params
        )
        
        print("✅ Sentiment Agent: Analysis completed")
        return result
    
    def _synthesize_sentiment_analysis(self, current_data: Dict[str, Any], 
                                    historical_data: Dict[str, Any], 
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize comprehensive sentiment analysis
        """
        sentiment_dist = current_data["sentiment_distribution"]
        
        # Determine overall sentiment
        if sentiment_dist["positive"] > 50:
            overall_sentiment = "highly_positive"
        elif sentiment_dist["positive"] > 30:
            overall_sentiment = "moderately_positive"
        else:
            overall_sentiment = "mixed"
        
        # Extract key themes from comments
        key_themes = self._extract_key_themes(current_data)
        
        # Platform analysis
        platform_insights = self._analyze_platform_distribution(current_data)
        
        return {
            "total_comments": current_data["total_comments"],
            "sentiment_breakdown": f"{sentiment_dist['positive']:.1f}% positive, {sentiment_dist['negative']:.1f}% negative, {sentiment_dist['neutral']:.1f}% neutral",
            "overall_sentiment": overall_sentiment,
            "key_themes": key_themes,
            "platform_insights": platform_insights,
            "engagement_metrics": {
                "average_engagement": current_data["average_engagement"],
                "high_engagement_posts": len([
                    c for c in current_data.get("comments", [])
                    if c.get("engagement", 0) > 1000
                ])
            },
            "historical_context": historical_data["insights"][:2],
            "recommendations": self._generate_sentiment_recommendations(sentiment_dist, key_themes)
        }
    
    def _extract_key_themes(self, data: Dict[str, Any]) -> List[str]:
        """Extract key themes from fan comments"""
        themes = []
        
        # Analyze post types
        post_types = data.get("post_types", [])
        for post_type in post_types:
            if post_type["count"] > 0:
                themes.append(f"{post_type['type']} ({post_type['count']} posts)")
        
        # Add common themes based on sentiment analysis
        themes.extend([
            "Player performance appreciation",
            "Team spirit and loyalty",
            "Game excitement and energy",
            "Social media engagement"
        ])
        
        return themes[:5]  # Return top 5 themes
    
    def _analyze_platform_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement by platform"""
        platforms = data.get("platform_distribution", {})
        
        # Calculate engagement rates
        platform_insights = {}
        for platform, count in platforms.items():
            platform_insights[platform] = {
                "post_count": count,
                "engagement_rate": "high" if count > 2 else "medium"
            }
        
        return platform_insights
    
    def _generate_sentiment_recommendations(self, sentiment_dist: Dict[str, float], 
                                          themes: List[str]) -> List[str]:
        """Generate actionable recommendations based on sentiment"""
        recommendations = []
        
        if sentiment_dist["positive"] > 60:
            recommendations.append("Capitalize on positive momentum with celebratory content")
            recommendations.append("Amplify fan testimonials and success stories")
        
        if sentiment_dist["negative"] > 20:
            recommendations.append("Address concerns through targeted communication")
            recommendations.append("Focus on positive developments to shift narrative")
        
        if any("player" in theme.lower() for theme in themes):
            recommendations.append("Create player-focused content and behind-the-scenes access")
        
        recommendations.append("Maintain consistent engagement across all platforms")
        
        return recommendations