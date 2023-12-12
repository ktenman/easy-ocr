from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from pydantic import BaseModel, ValidationError
import logging
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Set 'gpu' to False if you're not using a GPU

class ImageUploadRequest(BaseModel):
    uuid: str
    timestamp: str
    image: str

    def decode_image(self):
        # Decoding the base64 image
        return base64.b64decode(self.image.encode('utf-8'))

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_image(nparr):
    # Convert the buffer to an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Preprocessing steps like Gaussian blur, thresholding can be added here
    # For simplicity, we'll skip them in this example
    return gray

def perform_ocr(preprocessed_img):
    # Use EasyOCR to read the text from the image
    results = reader.readtext(preprocessed_img)
    # Join the text items to form the final result
    decoded_text = " ".join([result[1] for result in results])
    return decoded_text.strip() if decoded_text else "No text found"

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Log request excluding the image
        log_data = request.json.copy()
        log_data['image'] = 'image_data_not_logged'
        app.logger.info("Received request: %s", log_data)

        upload_request = ImageUploadRequest(**request.get_json())
        nparr = np.frombuffer(upload_request.decode_image(), np.uint8)
        preprocessed_img = preprocess_image(nparr)
        decoded_text = perform_ocr(preprocessed_img)

        return jsonify({
            "uuid": upload_request.uuid,
            "timestamp": upload_request.timestamp,
            "decoded_text": decoded_text
        })
    except ValidationError as e:
        app.logger.error("Validation Error: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error("Unhandled Exception: %s", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=61234)
