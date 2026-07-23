"""Locust load test: simulates users hitting the predict endpoint repeatedly."""
import os
from locust import HttpUser, task, between

# small sample image bytes reused for every simulated request
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample_cell.png")

class MalariaAPIUser(HttpUser):
    wait_time = between(1, 3)  # simulate realistic gaps between user requests

    @task
    def predict(self):
        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            files = {"file": ("sample_cell.png", f, "image/png")}
            self.client.post("/predict", files=files)

    @task(2)  # health check hit twice as often as predict, lighter load mix
    def health(self):
        self.client.get("/health")