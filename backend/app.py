from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import run_inference

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])


@app.route("/analyze", methods=["POST"])
def analyze():
    text_file = request.files.get("text_file")
    audio_file = request.files.get("audio_file")
    report_file = request.files.get("report_file")

    text_bytes = text_file.read() if text_file else None
    audio_bytes = audio_file.read() if audio_file else None
    report_bytes = report_file.read() if report_file else None

    try:
        result = run_inference(text_bytes, audio_bytes, report_bytes)
        return jsonify(result)
    except NotImplementedError:
        return jsonify({"error": "Model inference not yet implemented"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
