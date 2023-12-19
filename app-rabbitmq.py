import base64
import cv2
import easyocr
import logging
import numpy as np
import os
import pika
import requests
import signal
import sys
import threading
import time
from pydantic import BaseModel, ValidationError

IMAGE_REQUEST_QUEUE = 'picture-request-queue'
IMAGE_RESPONSE_QUEUE = 'picture-response-queue'
TEXT_REQUEST_QUEUE = 'text-request-queue'
TEXT_RESPONSE_QUEUE = 'text-response-queue'

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Set 'gpu' to False if you're not using a GPU

# Thread-local storage for UUID
thread_local_storage = threading.local()

class ImageUploadRequest(BaseModel):
    uuid: str
    image: str

    def decode_image(self):
        if self.image is not None:
            return base64.b64decode(self.image.encode('utf-8'))
        else:
            raise ValueError("Image data is None")

def preprocess_image(nparr):
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging.debug("Image preprocessed.")
    return gray

def perform_ocr(preprocessed_img):
    results = reader.readtext(preprocessed_img)
    decoded_text = " ".join([result[1] for result in results])
    logging.debug("OCR performed on the image.")
    return decoded_text.strip() if decoded_text else "No text found"

def publish_result(channel, extracted_text):
    try:
        channel.basic_publish(
            exchange='',
            routing_key=IMAGE_RESPONSE_QUEUE,
            body=f"{thread_local_storage.uuid}:{extracted_text}"
        )
        logging.debug(f"Published result to '{IMAGE_RESPONSE_QUEUE}'")
    except Exception as e:
        logging.error(f"Failed to publish result: {e}")

def handle_message(ch, method, properties, body):
    try:
        start_time = time.perf_counter()

        message_data = body.decode('utf-8')
        uuid, encoded_image = message_data.split(':', 1)

        thread_local_storage.uuid = uuid

        logging.info("Received message")

        upload_request = ImageUploadRequest(uuid=uuid, image=encoded_image)
        nparr = np.frombuffer(upload_request.decode_image(), np.uint8)
        preprocessed_img = preprocess_image(nparr)
        extracted_text = perform_ocr(preprocessed_img)

        logging.info(f"OCR result: '{extracted_text}'")

        publish_result(ch, extracted_text)

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        logging.info(f"Processed and published OCR result in {processing_time:.3f} seconds")

    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except ValidationError as e:
        logging.error(f"Validation Error: {e}")
    except Exception as e:
        logging.error(f"Unhandled Exception: {e}")
    finally:
        del thread_local_storage.uuid


def make_api_request(prompt, timeout_seconds=60):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "dolphin-mixtral",
        "prompt": prompt,
        "stream": False  # Request not to stream the response
    }

    try:
        response = requests.post(url, json=data, timeout=timeout_seconds)
        response.raise_for_status()
        json_response = response.json()
        logging.debug(f"API response: {json_response}")
        return json_response

    except requests.exceptions.Timeout:
        logging.error("The request timed out")
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {e}")

def handle_text_request(ch, method, properties, body):
    try:
        message_data = body.decode('utf-8')
        uuid, prompt = message_data.split(':', 1)

        thread_local_storage.uuid = uuid
        logging.info("Received text request message")

        api_response = make_api_request(prompt)
        full_response = api_response.get("response", "")

        # Strip leading spaces from the final response
        full_response = full_response.lstrip()

        logging.debug(f"Full response: {full_response}")
        publish_result_to_text_queue(ch, uuid, full_response)

    except Exception as e:
        logging.error(f"Unhandled Exception in text request handling: {e}")
    finally:
        del thread_local_storage.uuid

# Publish result to the text-response-queue
def publish_result_to_text_queue(channel, uuid, text):
    try:
        message = f"{uuid}:{text}"
        channel.basic_publish(
            exchange='',
            routing_key=TEXT_RESPONSE_QUEUE,
            body=message
        )
        logging.debug(f"Published text response to '{TEXT_RESPONSE_QUEUE}'")
    except Exception as e:
        logging.error(f"Failed to publish text response: {e}")

def setup_rabbitmq_connection():
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_USER'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='tenman.ee', credentials=credentials))
    return connection.channel()

def setup_queue(channel):
    channel.queue_declare(queue=IMAGE_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=IMAGE_RESPONSE_QUEUE, durable=True)
    channel.queue_declare(queue=TEXT_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=TEXT_RESPONSE_QUEUE, durable=True)

def subscribe_to_queue(channel):
    channel.basic_consume(queue=IMAGE_REQUEST_QUEUE, on_message_callback=handle_message, auto_ack=True)
    channel.basic_consume(queue=TEXT_REQUEST_QUEUE, on_message_callback=handle_text_request, auto_ack=True)

class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        ms = int(record.msecs)
        return f"{created}.{ms:03d}"

class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super(CustomLogRecord, self).__init__(*args, **kwargs)
        uuid = getattr(thread_local_storage, 'uuid', None)
        self.uuid = f"[UUID: {uuid}] " if uuid else ""

def setup_logging():
    logging.setLogRecordFactory(CustomLogRecord)
    logging_format = "%(asctime)s - %(levelname)s - %(uuid)s%(message)s"
    formatter = MillisecondFormatter(logging_format)
    logging.basicConfig(level=logging.DEBUG, format=logging_format)
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    setup_logging()
    logging.getLogger("pika").setLevel(logging.INFO)
    channel = setup_rabbitmq_connection()
    setup_queue(channel)
    subscribe_to_queue(channel)

    channel.start_consuming()

if __name__ == '__main__':
    main()
