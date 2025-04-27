# from django.shortcuts import render

from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .models import Driver
from ml_models.face_recognition import recognize_face
import cv2
import numpy as np
import base64
import json

# View to retrieve driver information by license number
class DriverView(APIView):
    def get(self, request, license_number):
        try:
            driver = Driver.objects.get(license_number=license_number)
            data = {
                'name': driver.name,
                'license_number': driver.license_number,
                'license_status': driver.license_status,
                'violation_history': driver.violation_history,
                'penalties': driver.penalties,
            }
            return Response(data)
        except Driver.DoesNotExist:
            return Response({'error': 'Driver not found'}, status=404)

# View for face recognition
class FaceRecognitionView(APIView):
    parser_classes = [MultiPartParser]
    def get(self, request):
        return Response({"message": "Face recognized"})

    def post(self, request):
        # Get the image file from the request
        image_file = request.FILES['image']
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)
        # new line added for debugging
        # print("FILES received:", request.FILES)
        # print("DATA received:", request.data)
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return JsonResponse({'error': 'Invalid image format'}, status=400)
        print("Image shape:", image.shape)
        

        # Perform face recognition
        license_number, processed_image = recognize_face(image)

        # Encode the processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        if license_number:
            return JsonResponse({
                'license_number': license_number,
                'processed_image': encoded_image,
            })
        else:
            return JsonResponse({'error': 'No face recognized',
            'processed_image': encoded_image,  # still show face with red rectangle
        }, status=400)