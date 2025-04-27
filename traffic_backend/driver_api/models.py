from django.db import models

# Create your models here.

class Driver(models.Model):
    name = models.CharField(max_length=255)
    license_number = models.CharField(max_length=20, unique=True)
    license_status = models.CharField(max_length=20, default='active')
    violation_history = models.TextField(blank=True, null=True)
    penalties = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    face_embedding = models.TextField(blank=True, null=True)  # Store FaceNet embeddings as JSON

    def __str__(self):
        return self.name