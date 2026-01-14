import os
import time
from typing import List, Dict
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

# Model price table (price per 1000 tokens, unit: USD)
MODEL_PRICES = {

}

class APIClient:
    """API Client"""
    
    # Class-level total cost tracking
    _total_cost = 0.0
    
    def __init__(self, model_name: str = "google/gemini-2.5-pro-preview", API_Type: str = "openrouter"):
        self.model_name = model_name
        self.API_Type = API_Type
        if API_Type == "openai":
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif API_Type == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_KEY"),
            )
        else:
            raise ValueError(f"Unsupported API type: {API_Type}")
        logger.info(f"Successfully initialized client, using model: {self.model_name}")
        # Instance-level cost tracking
        self.instance_cost = 0.0
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_tokens: int = 8192, temperature: float = 0.1,
                         retry_count: int = 3, n: int = 1) -> str:
        """Generate response"""
        if self.API_Type == "openai":   
            for attempt in range(retry_count):
                try:
                    if self.model_name in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_completion_tokens=max_tokens,
                            n=n
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=max_tokens,
                            n=n
                        )
                    # Calculate and record cost
                    self._calculate_and_add_cost(completion)
                    
                    if n == 1:
                        response = completion.choices[0].message.content
                    else:
                        response = [choice.message.content for choice in completion.choices]
                    return response
                    
                except Exception as e:
                    logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                    if attempt == retry_count - 1:
                        logger.error(f"API call finally failed: {e}")
                        return "$ERROR$"
                    time.sleep(2 ** attempt)  # Exponential backoff
        elif self.API_Type == "openrouter":
            for attempt in range(retry_count):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=n
                    )
                    # Calculate and record cost
                    self._calculate_and_add_cost(completion)
                    
                    if n == 1:
                        response = completion.choices[0].message.content
                    else:
                        response = [choice.message.content for choice in completion.choices]
                    return response

                except Exception as e:
                    logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                    if attempt == retry_count - 1:
                        logger.error(f"API call finally failed: {e}")
                        return "$ERROR$"
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return "$ERROR$"
    
    def _calculate_and_add_cost(self, completion):
        """Calculate and accumulate API call cost"""
        try:
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                
                # Get model price
                model_price = self._get_model_price()
                if model_price:
                    input_cost = (input_tokens / 1000.0) * model_price["input"]
                    output_cost = (output_tokens / 1000.0) * model_price["output"]
                    total_cost = input_cost + output_cost
                    
                    # Add to class-level and instance-level costs
                    APIClient._total_cost += total_cost
                    self.instance_cost += total_cost
                    
                    logger.info(
                        f"API call cost - Model: {self.model_name}, "
                        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, "
                        f"Current cost: ${total_cost:.6f}, Instance total cost: ${self.instance_cost:.6f}, "
                        f"Global total cost: ${APIClient._total_cost:.6f}"
                    )
                else:
                    logger.warning(f"Price information not found for model {self.model_name}")
            else:
                logger.warning("No usage information in API response, unable to calculate cost")
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
    
    def _get_model_price(self):
        """Get model price information"""
        # Direct model name matching
        if self.model_name in MODEL_PRICES:
            return MODEL_PRICES[self.model_name]
        
        # For OpenRouter, try to match base model names
        if self.API_Type == "openrouter":
            for model_key in MODEL_PRICES.keys():
                if model_key in self.model_name or self.model_name in model_key:
                    return MODEL_PRICES[model_key]
        
        return None
    
    @classmethod
    def get_total_cost(cls):
        """Get total cost of all API calls"""
        return cls._total_cost
    
    @classmethod
    def reset_total_cost(cls):
        """Reset total cost counter"""
        cls._total_cost = 0.0
        logger.info("Global total cost counter has been reset")
    
    def get_instance_cost(self):
        """Get total cost of current instance"""
        return self.instance_cost
    
    def reset_instance_cost(self):
        """Reset cost counter of current instance"""
        self.instance_cost = 0.0
        logger.info(f"Instance cost counter has been reset, model: {self.model_name}")
    
    def get_cost_summary(self):
        """Get cost summary information"""
        return {
            "model_name": self.model_name,
            "api_type": self.API_Type,
            "instance_cost": self.instance_cost,
            "total_cost": APIClient._total_cost,
            "cost_currency": "USD"
        }


