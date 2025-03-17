from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
CORS(app)

# Load OpenAI API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/generate", methods=["POST"])
def generate_design():
    try:
        text_prompt = request.json["text"]
        print("Received request:", text_prompt)

        # NOTE OpenAI Image Generation
        response = client.images.generate(
            model="dall-e-3",  # Correct model name
            prompt=text_prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url  # NOTE Proper way to access image URL

        #  GPT-4o Text Generation creates teh content for teh description from the image *** NOTE i comment it out becuase is not need it for this DEMO
        # completion = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[{"role": "user", "content": f"Describe this AI-generated architecture: {text_prompt}"}]
        # )
        # print("GPT-4o Response:", completion)

        # description = completion.choices[0].message.content  #  The way to access response NOTE

        return jsonify({"image_url": image_url,
                        #  "description": description
                         })

    except Exception as e:
        print("Error in /generate:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        image_url = request.json["image_url"]
        print("Received Image URL:", image_url)  # Debugging print

        # NOTE OpenAI Image Analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url}}  # Correct format

                    ],
                }
            ]
        )
        print("GPT-4o Response:", response)

        analysis = response.choices[0].message.content  # NOTE way to access response

        return jsonify({"analysis": analysis})

    except Exception as e:
        print("Error in /analyze-image:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)