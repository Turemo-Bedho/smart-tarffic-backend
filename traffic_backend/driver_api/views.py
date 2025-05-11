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














# #  code for generating traffic reports
# import json
# import logging
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# # from .reports.services.report_service import TrafficReportGenerator

# from reports.services.report_service import TrafficReportGenerator
# logger = logging.getLogger(__name__)



# @csrf_exempt
# def generate_traffic_report(request):
#     """Handles report generation requests."""
#     if request.method != "POST":
#         return JsonResponse({"error": "Only POST requests allowed"}, status=405)

#     try:
        
#         data = json.loads(request.body)
      
#         query_text = data.get("query_text", "")
       

#         if not query_text:
#             return JsonResponse({"error": "Missing query_text parameter"}, status=400)
       
#         report_generator = TrafficReportGenerator()
       
#         result = report_generator.generate_report(query_text)
#         result['pdf_url'] = request.build_absolute_uri(result['pdf_url'])
#         return JsonResponse(result)

#     except json.JSONDecodeError:
#         logger.error("Invalid JSON format received in request")
#         return JsonResponse({"error": "Invalid JSON format"}, status=400)

#     except Exception as e:
#         logger.exception(f"Unexpected error: {e}")
#         return JsonResponse({"error": str(e)}, status=500)



from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
import logging
from django.views.decorators.csrf import csrf_exempt


logger = logging.getLogger(__name__)

@csrf_exempt
def generate_traffic_report(request):
    """API endpoint for report generation"""
    try:
        # Parse input
        try:
            data = json.loads(request.body)
            query_text = data.get('query', '').strip()
            if not query_text:
                return JsonResponse(
                    {"status": "error", "error": "Empty query"}, 
                    status=400
                )
        except json.JSONDecodeError:
            return JsonResponse(
                {"status": "error", "error": "Invalid JSON"}, 
                status=400
            )
        
        # Generate report
        from reports.services.report_service import TrafficReportGenerator
        generator = TrafficReportGenerator()
        result = generator.generate_report(query_text)
        
        # Build full PDF URL
        if result['status'] == 'success':
            result['pdf_url'] = request.build_absolute_uri(result['pdf_url'])
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.exception("Report endpoint failed")
        return JsonResponse(
            {"status": "error", "error": "Internal server error"},
            status=500
        )