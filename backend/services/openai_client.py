from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIService:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_image(self, prompt, size="1024x1024"):
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size
            )
            return response.data[0].url
        except Exception as e:
            print("Error generating image:", e)
            return None

    def analyze_image(self, image_url):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error analyzing image:", e)
            return None

    #  Explain Prediction
    def explain_prediction(self, predictions: dict):
        """
        Accepts a dictionary of predicted conditions and confidence scores,
        returns an explanation string for all detected conditions.
        """
        try:
            if not predictions:
                return "No abnormal findings detected in the X-ray."

            conditions = list(predictions.keys())
            joined = ", ".join(conditions)
            message = (
                f"The following conditions were detected in the chest X-ray: {joined}.\n"
                "Please provide a brief medical explanation for each one."
            )

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print("Error in explanation generation:", e)
            return "Explanation could not be generated."