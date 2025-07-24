from flask import Flask, request, jsonify
from yolov8_detect import detect_objects

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image_data = image_file.read()

    # Perform object detection
    results = detect_objects(image_data)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

