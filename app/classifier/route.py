import io
import base64
import torch

from flask import Blueprint, request, jsonify
from PIL import Image

from app.classifier.classifier import (
    get_model,
    predict_image
)

from app.blurry.blurry_service import de_blur
from app.dark.dark_service import de_dark
from app.foggy.foggy_service import de_fog
from app.old.old_service import de_old


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WEIGHTS_PATH = r"C:\\NeuroVisionCombined4\\models\\classifier.pth"

classifier = Blueprint("classifier", __name__)

model = get_model(WEIGHTS_PATH, device)


ROUTE_TO_SERVICE = {
    "blurry": de_blur,
    "dark": de_dark,
    "foggy": de_fog,
    "damaged": de_old
}


def image_to_base64(pil_image):

    buffer = io.BytesIO()

    pil_image.save(buffer, format="PNG")

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return encoded


@classifier.route("/predict", methods=["POST"])
def predict():

    try:

        if not model:
            return jsonify({
                "message": "Model not loaded",
                "image": None
            }), 500

        if "image" not in request.files:
            return jsonify({
                "message": "No image provided",
                "image": None
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "message": "Empty filename",
                "image": None
            }), 400

        if not file.mimetype.startswith("image/"):
            return jsonify({
                "message": "File must be an image",
                "image": None
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
                "image": None
            }), 500

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        current_image = pil_image

        applied_routes = []

        # No Restoration
        if "normal" in routes:

            return jsonify({
                "message": "No restoration needed",
                "image": image_to_base64(current_image)
            }), 200

        # Uncertain
        if "uncertain" in routes:

            return jsonify({
                "message": "Uncertain degradation detected",
                "image": None
            }), 200

        # Sequential Pipeline
        for route in routes:

            if route in ROUTE_TO_SERVICE:

                service_function = ROUTE_TO_SERVICE[route]

                current_image = service_function(current_image)

                applied_routes.append(route)

        return jsonify({
            "message": f"Applied: {', '.join(applied_routes)}",
            "image": image_to_base64(current_image)
        }), 200

    except Exception as e:

        return jsonify({
            "message": str(e),
            "image": None
        }), 500