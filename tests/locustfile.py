import os

from locust import HttpUser, between, task


class FastAPIUser(HttpUser):
    wait_time = between(1, 3)
    base_url = os.getenv("MYENDPOINT", "http://localhost:8000")  # Default to local API

    @task
    def classify_image(self):
        with open("data/raw/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg", "rb") as image_file:
            self.client.post(f"{self.base_url}/classify/", files={"file": ("test_image.jpg", image_file, "image/jpeg")})

    @task
    def post_data(self):
        data_payload = {"x": "dog", "probabilities": [0.1, 0.2, 0.3, 0.05, 0.1, 0.15, 0.1, 0.05, 0.02, 0.03]}
        self.client.post(f"{self.base_url}/post_data", json=data_payload)
