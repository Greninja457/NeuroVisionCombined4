import io
import time
from minio import Minio

minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)

BUCKET_NAME = "neuro-vision"


def upload_to_minio(pil_image, user_id):

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    object_name = (
        f"users/{user_id}/enhanced/"
        f"enhanced_{int(time.time())}.png"
    )

    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="image/png"
    )

    return object_name