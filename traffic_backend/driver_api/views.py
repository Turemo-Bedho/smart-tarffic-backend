# from django.shortcuts import render

import json
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.viewsets import ModelViewSet
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework.decorators import action
from .models import Driver
from .face_recognition import recognize_face_one
import cv2
import numpy as np
from .utils import process_face
from .validators import validate_embedding
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import uuid
import os
from rest_framework import viewsets
from .models import Driver, Officer, Address
from .serializers import DriverSerializer, OfficerSerializer, AddressSerializer

class DriverView(viewsets.ModelViewSet):
    queryset = Driver.objects.all()
    serializer_class = DriverSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        image_file = self.request.FILES['profile_image']
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

            
        
        try:
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            embedding, aligned_face = process_face(image)
            validate_embedding(embedding)
        except DjangoValidationError as e:
            raise DRFValidationError({'embedding': e.messages})
        except Exception as e:
            raise DRFValidationError({'embedding': "The image could not be processed."})
        embedding_json = json.dumps(embedding.tolist())
        serializer.save(embedding=embedding_json)
        return super().perform_create(serializer)
    
    @action(detail=False, methods=['post'], url_path='identify')
    def identify_driver(self, request):
        try:
            image_file = request.FILES['image']
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            license_number, processed_image = recognize_face_one(image)

            # Encode the processed image to base64
            _, buffer = cv2.imencode('.jpg', processed_image)
            filename = f"processed_images/{uuid.uuid4().hex}.jpg"
            path = default_storage.save(filename, ContentFile(buffer.tobytes()))
            image_url = os.path.join(settings.MEDIA_URL, path)

            if license_number:
                driver = Driver.objects.get(license_number=license_number)
                serializer = self.get_serializer(driver)
                return Response({
                    'driver': serializer.data,
                    'processed_image_url': request.build_absolute_uri(image_url)
                })

            else:
                raise DRFValidationError({'Identification': "Couldn't identify the driver."})
        
        except Exception as e:
            print("Error during face recognition:", str(e))
            raise DRFValidationError({'Identification': "Couldn't identify the driver."})




class OfficerView(viewsets.ModelViewSet):
    queryset = Officer.objects.all()
    serializer_class = OfficerSerializer

class AddressView(viewsets.ModelViewSet):
    queryset = Address.objects.all()
    serializer_class = AddressSerializer
