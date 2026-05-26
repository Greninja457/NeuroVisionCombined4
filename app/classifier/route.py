import io
import torch

from PIL import Image
from flask import Blueprint, request, jsonify
from utils.minio import upload_to_minio
from app.classifier.classifier import ( get_model, predict_image)
from app.blurry.blurry_service import de_blur
from app.dark.dark_service import de_dark
from app.foggy.foggy_service import de_fog
from app.old.old_service import de_old


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = r"C:\\NeuroVisionCombined4\\models\\classifier.pth"

classifier = Blueprint("classifier", __name__)

model = get_model(WEIGHTS_PATH, device)


ROUTE_TO_SERVICE = {
    "blurry": de_blur,
    "dark": de_dark,
    "foggy": de_fog,
    "damaged": de_old
}

@classifier.route("/predict", methods=["POST"])
def predict():

    try:

        if not model:
            return jsonify({
                "message": "Model not loaded",
                "enhancements": []
            }), 500

        if "image" not in request.files:
            return jsonify({
                "message": "No image provided",
                "enhancements": []
            }), 400

        if "id" not in request.form:
            return jsonify({
                "message": "User id missing",
                "enhancements": []
            }), 400

        user_id = request.form["id"]

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "message": "Empty filename",
                "enhancements": []
            }), 400

        if not file.mimetype.startswith("image/"):
            return jsonify({
                "message": "File must be an image",
                "enhancements": []
            }), 400

        image_bytes = file.read()

        image_stream = io.BytesIO(image_bytes)

        prediction = predict_image(
            model=model,
            image_bytes=image_stream,
            device=device
        )

        routes = prediction.get("route", [])

        if not routes:
            return jsonify({
                "message": "No route generated",
                "enhancements": []
            }), 500

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        current_image = pil_image

        applied_routes = []

        if "normal" in routes:

            upload_to_minio(
                current_image,
                user_id
            )

            return jsonify({
                "message": "No restoration needed",
                "enhancements": []
            }), 200

        if "uncertain" in routes:

            return jsonify({
                "message": "Uncertain degradation detected",
                "enhancements": []
            }), 200

        for route in routes:

            if route in ROUTE_TO_SERVICE:

                service_function = ROUTE_TO_SERVICE[route]

                current_image = service_function(current_image)

                applied_routes.append(route)

        upload_to_minio(
            current_image,
            user_id
        )

        return jsonify({
            "message": "Enhancement completed",
            "enhancements": applied_routes
        }), 200

    except Exception as e:

        return jsonify({
            "message": str(e),
            "enhancements": []
        }), 500