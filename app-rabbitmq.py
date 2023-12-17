import base64
import cv2
import easyocr
import logging
import numpy as np
import os
import sys
import signal
import pika
import threading
import time
from pydantic import BaseModel, ValidationError

IMAGE_REQUEST_QUEUE = 'picture-request-queue'
IMAGE_RESPONSE_QUEUE = 'picture-response-queue'

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

def setup_rabbitmq_connection():
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_USER'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='tenman.ee', credentials=credentials))
    return connection.channel()

def setup_queue(channel):
    channel.queue_declare(queue=IMAGE_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=IMAGE_RESPONSE_QUEUE, durable=True)

def subscribe_to_queue(channel):
    channel.basic_consume(queue=IMAGE_REQUEST_QUEUE, on_message_callback=handle_message, auto_ack=True)
    channel.start_consuming()

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

def main():
    setup_logging()
    logging.getLogger("pika").setLevel(logging.INFO)

    channel = setup_rabbitmq_connection()
    setup_queue(channel)
    subscribe_to_queue(channel)

if __name__ == '__main__':
    main()
