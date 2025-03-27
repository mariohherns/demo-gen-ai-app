from flask import Blueprint, request, jsonify
from PIL import Image, UnidentifiedImageError

analyze_bp = Blueprint("analyze", __name__)


def register_analyze_routes(app, fracture_model, openai_service):

    @analyze_bp.route("/full-analyze", methods=["POST"])
    def full_analyze():
        try:
            # Check if image part exists in the request
            if "image" not in request.files:
                raise ValueError("No 'image' field in request.files")

            file = request.files["image"]
            if file.filename == "":
                raise ValueError("No file selected")

            try:
                image = Image.open(file).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image")

            # Run inference
            label, confidence = fracture_model.predict(image)

            # Explanation from GPT (optional)
            explanation = openai_service.explain_prediction(label)

            return jsonify({
                "prediction": label,
                "confidence": round(confidence, 3),
                "explanation": explanation
            })

        except Exception as e:
            print("Error in /full-analyze:", str(e))  # Debug log
            return jsonify({"error": str(e)}), 500

    app.register_blueprint(analyze_bp)
    