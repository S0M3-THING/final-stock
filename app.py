from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cheatsheet import load_resnet, load_clip, extract_features_resnet, extract_features_clip, buy_sell_mapping
import gunicorn

app = Flask(__name__, static_folder="static", template_folder="templates")


resnet_model = load_resnet()
clip_model, clip_preprocess, clip_device = load_clip()


root_dir = "./cheatsheet"
database_images = list(buy_sell_mapping.keys())
database_paths = [os.path.join(root_dir, img) for img in database_images]

database_features_resnet = np.array([extract_features_resnet(resnet_model, img) for img in database_paths])
database_features_clip = np.array([extract_features_clip(clip_model, clip_preprocess, clip_device, img) for img in database_paths])

@app.route("/")
def index():
    return send_from_directory("templates", "frontend.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files["image"]
    image_path = "uploaded_image.jpg"
    image.save(image_path)


    input_features_resnet = extract_features_resnet(resnet_model, image_path)
    input_features_clip = extract_features_clip(clip_model, clip_preprocess, clip_device, image_path)


    similarities_resnet = cosine_similarity([input_features_resnet], database_features_resnet)
    closest_resnet_idx = np.argmax(similarities_resnet)
    resnet_match = database_images[closest_resnet_idx]
    confidence_resnet = float(similarities_resnet[0, closest_resnet_idx] * 100)

    similarities_clip = cosine_similarity([input_features_clip], database_features_clip)
    closest_clip_idx = np.argmax(similarities_clip)
    clip_match = database_images[closest_clip_idx]
    confidence_clip = float(similarities_clip[0, closest_clip_idx] * 100)


    resnet_decision = buy_sell_mapping[resnet_match]
    clip_decision = buy_sell_mapping[clip_match]

    final_decision = resnet_decision if resnet_decision == clip_decision else "uncertain"

    return jsonify({
        "resnet_match": resnet_match,
        "resnet_confidence": confidence_resnet,
        "resnet_decision": resnet_decision,
        "clip_match": clip_match,
        "clip_confidence": confidence_clip,
        "clip_decision": clip_decision,
        "final_decision": final_decision
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
