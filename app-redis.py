import base64
import cv2
import easyocr
import logging
import numpy as np
import redis
import threading
import time
from pydantic import BaseModel, ValidationError

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Set 'gpu' to False if you're not using a GPU

# Thread-local storage for UUID
thread_local_storage = threading.local()

class ImageUploadRequest(BaseModel):
    uuid: str
    image: str

    def decode_image(self):
        return base64.b64decode(self.image.encode('utf-8'))

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

def publish_result(redis_client, decoded_text):
    try:
        redis_client.publish('image-response-queue', f"{thread_local_storage.uuid}:{decoded_text}")
        logging.debug("Published result to 'response-queue'")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis Connection Error while publishing: {e}")

def handle_message(redis_client, message):
    try:
        start_time = time.perf_counter()

        message_data = message['data'].decode('utf-8')
        uuid, encoded_image = message_data.split(':', 1)

        thread_local_storage.uuid = uuid

        logging.info("Received message")

        upload_request = ImageUploadRequest(uuid=uuid, image=encoded_image)
        nparr = np.frombuffer(upload_request.decode_image(), np.uint8)
        preprocessed_img = preprocess_image(nparr)
        decoded_text = perform_ocr(preprocessed_img)

        logging.info(f"OCR result: {decoded_text}")

        publish_result(redis_client, decoded_text)

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        logging.info(f"Processed and published OCR result in {processing_time:.3f} seconds")

    except ValidationError as e:
        logging.error(f"Validation Error: {e}")
    except Exception as e:
        logging.error(f"Unhandled Exception: {e}")
    finally:
        del thread_local_storage.uuid

def create_redis_client():
    return redis.Redis(host='tenman.ee', port=6379, db=0, socket_timeout=5)

def subscribe_to_queue(redis_client):
    pubsub = redis_client.pubsub()

    def message_handler(message):
        handle_message(redis_client, message)

    pubsub.subscribe(**{'image-request-queue': message_handler})
    return pubsub.run_in_thread(sleep_time=0.001)

def check_redis_connection(client):
    try:
        return client.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        return False

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

def main():
    setup_logging()

    redis_client = None
    timeout_seconds = 10
    is_subscribed = False
    is_first_connection = True

    while True:
        try:
            if not redis_client or not check_redis_connection(redis_client):
                redis_client = create_redis_client()
                is_subscribed = False

            if not is_subscribed:
                subscribe_to_queue(redis_client)
                is_subscribed = True

                if is_first_connection:
                    logging.info("Successfully connected and subscribed to Redis.")
                    is_first_connection = False
                else:
                    logging.info("Successfully reconnected and subscribed to Redis.")

        except redis.exceptions.ConnectionError as e:
            logging.error(f"Redis Connection Error: {e}. Reconnecting in {timeout_seconds} seconds...")
            redis_client = None
            time.sleep(timeout_seconds)
        except Exception as e:
            logging.error(f"Unhandled Exception: {e}. Reconnecting in {timeout_seconds} seconds...")
            redis_client = None
            time.sleep(timeout_seconds)

        time.sleep(1)

if __name__ == '__main__':
    main()
