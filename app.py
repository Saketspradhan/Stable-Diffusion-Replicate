from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from datetime import datetime
import replicate
import requests
import json
import os


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/StableDiffusion/<string:prompt>")
@cross_origin()

def StableDiffusion(prompt):
    os.environ['REPLICATE_API_TOKEN'] = "29e367c8558128b088e6d0716b27b78febdcecc1"
    model = replicate.models.get("stability-ai/stable-diffusion")
    output = model.predict(prompt=prompt, width=1024, height=768, num_inference_steps=100) # seed -> randomize
    return [output[0], prompt]


if __name__ == "__main__":
    app.run(debug=False)