# Standard Library Imports
import json
import uuid
from io import BytesIO
import io

# Third-Party Library Imports
import boto3
from botocore.exceptions import ClientError
import cv2
from django.conf import settings
from django.http import JsonResponse # Note: This is a Django specific import, so it's a bit of a grey area. Keeping it with Django-related imports
from django.views.decorators.csrf import csrf_exempt # Django specific
import numpy as np
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import action, api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from driver_api.plate_recogntion import EthiopianPlateRecognizer

# Local Application/Project Imports
from .models import Driver, Officer, Address, Violation, ViolationType, Vehicle
from .serializers import DriverSerializer, OfficerSerializer, AddressSerializer, ViolationSerializer, ViolationTypeSerializer, VehicleSerializer


class DriverView(viewsets.ModelViewSet):
    queryset = Driver.objects.all()
    serializer_class = DriverSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        try:
            s3_client = boto3.client('s3')
            rekognition_client = boto3.client('rekognition')
            
            image = self.request.FILES['profile_image']
            file_extension = image.name.split('.')[-1]
            unique_filename = f"profiles/{uuid.uuid4()}.{file_extension}"
            
            s3_client.upload_fileobj(
                image,
                settings.AWS_STORAGE_BUCKET_NAME,
                unique_filename,
                ExtraArgs={'ContentType': image.content_type}
            )
            
            
            collection_id = settings.AWS_REKOGNITION_COLLECTION_ID
            response = rekognition_client.index_faces(
                CollectionId=collection_id,
                Image={'S3Object': {
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Name': unique_filename
                }},
                ExternalImageId=str(serializer.validated_data['license_number']),
                DetectionAttributes=['ALL']
            )
            
            if not response['FaceRecords']:
                return Response(
                    {'error': 'No face detected in the image'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            
            serializer.validated_data['profile_image'] = unique_filename
            return super().perform_create(serializer)
            
        except ClientError as e:
            return Response(
                {'error': 'Failed to process image with AWS services'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
    
    
    @action(detail=False, methods=['post'], url_path='identify')
    def identify_driver(self, request):
        try:
            s3_client = boto3.client('s3')
            rekognition_client = boto3.client('rekognition')
            
            # Upload temporary image to S3
            image = request.FILES['image']
            temp_filename = f"temp/{uuid.uuid4()}.{image.name.split('.')[-1]}"
            
            s3_client.upload_fileobj(
                image,
                settings.AWS_STORAGE_BUCKET_NAME,
                temp_filename,
                ExtraArgs={'ContentType': image.content_type}
            )
            
            # Search for matching faces
            response = rekognition_client.search_faces_by_image(
                CollectionId=settings.AWS_REKOGNITION_COLLECTION_ID,
                Image={'S3Object': {
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Name': temp_filename
                }},
                MaxFaces=1,
                FaceMatchThreshold=90
            )
            
            # Clean up temporary file
            s3_client.delete_object(
                Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                Key=temp_filename
            )
            
            if not response['FaceMatches']:
                return Response(
                    {'error': 'No matching driver found'},
                    status=status.HTTP_404_NOT_FOUND
                )
                
            # Get driver details
            license_number = response['FaceMatches'][0]['Face']['ExternalImageId']
            try:
                driver = Driver.objects.get(license_number=license_number)
                return Response(
                    DriverSerializer(driver).data,
                    status=status.HTTP_200_OK
                )
            except Driver.DoesNotExist:
                return Response(
                    {'error': 'Driver not found in database'},
                    status=status.HTTP_404_NOT_FOUND
                )
                
        except ClientError as e:
            return Response(
                {'error': 'Failed to process image with AWS services'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




class OfficerView(viewsets.ModelViewSet):
    queryset = Officer.objects.all()
    serializer_class = OfficerSerializer

class AddressView(viewsets.ModelViewSet):
    queryset = Address.objects.all()
    serializer_class = AddressSerializer


class ViolationView(viewsets.ModelViewSet):
    queryset = Violation.objects.all()
    serializer_class = ViolationSerializer

    def get_serializer_context(self):
        return {
            "user_id": self.request.user.id,
        }
    

class ViolationTypeView(viewsets.ModelViewSet):
    queryset = ViolationType.objects.all()
    serializer_class = ViolationTypeSerializer


class VehicleView(viewsets.ModelViewSet):
    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer



# @api_view(['POST'])
# def check_point(request):
#     response_data = {
#         'success': False,
#         'plate_image_url': None,
#         'text': "Doesn't have a driver licence.",
#     }
#     try:
#         s3_client = boto3.client('s3')
#         rekognition_client = boto3.client('rekognition')
        
#         # Read image data into memory
#         image = request.FILES['image']
#         image_data = image.read()  # Read once
#         image_buffer = BytesIO(image_data)  # For S3 upload
#         file_extension = image.name.split('.')[-1]
#         content_type = image.content_type
        
#         # Upload temporary image to S3
#         temp_filename = f"temp/{uuid.uuid4()}.{file_extension}"
        
#         s3_client.upload_fileobj(
#             image_buffer,
#             settings.AWS_STORAGE_BUCKET_NAME,
#             temp_filename,
#             ExtraArgs={'ContentType': image.content_type}
#         )
        
#         # Search for matching faces
#         response = rekognition_client.search_faces_by_image(
#             CollectionId=settings.AWS_REKOGNITION_COLLECTION_ID,
#             Image={'S3Object': {
#                 'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
#                 'Name': temp_filename
#             }},
#             MaxFaces=1,
#             FaceMatchThreshold=90
#         )
        
#         # Clean up temporary file
#         s3_client.delete_object(
#             Bucket=settings.AWS_STORAGE_BUCKET_NAME,
#             Key=temp_filename
#         )
        
#         if not response['FaceMatches']:
#             image_array = np.frombuffer(image_data, np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             if image is None or image.size == 0:
#                 response_data['debug_info'].append("Invalid image format")
#                 return Response({'error': 'Invalid image format'}, status=400)

#             try:
#                 recognizer = EthiopianPlateRecognizer(debug_mode=True)
#                 plates = recognizer.process_image(image)

#                 if not plates.flags['C_CONTIGUOUS']:
#                     plates = plates.copy()

#                 temp_filename = f"plates/{uuid.uuid4()}.{file_extension}"

#                 # Convert the numpy array to bytes and create a file-like object
#                 is_success, buffer = cv2.imencode(f".{file_extension}", plates)
#                 if not is_success:
#                     raise ValueError("Could not encode image")

#                 bytes_io = io.BytesIO(buffer)

#                 s3_client.upload_fileobj(
#                     bytes_io,  # This is a file-like object that implements read()
#                     settings.AWS_STORAGE_BUCKET_NAME,
#                     temp_filename,
#                     ExtraArgs={'ContentType': content_type}
#                 )


#                 response_data['plate_image_url'] = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{temp_filename}"

        
#             except Exception as e:
#                 return Response(
#                     {'error': 'Failed to recognize plate'},
#                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
#                 )

#             # Point to push notification to the officer.
#             return Response(response_data)
            
#         # Get driver details
#         license_number = response['FaceMatches'][0]['Face']['ExternalImageId']
#         try:
#             driver = Driver.objects.get(license_number=license_number)
#             return Response(
#                 DriverSerializer(driver).data,
#                 status=status.HTTP_200_OK
#             )
#         except Driver.DoesNotExist:
#             return Response(
#                 {'error': 'Driver not found in database'},
#                 status=status.HTTP_404_NOT_FOUND
#             )
            
#     except ClientError as e:
#         return Response(
#             {'error': 'Failed to process the image.'},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )
#     except Exception as e:
#         return Response(
#             {'error': 'Internal server error'},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )










@api_view(['POST'])
def check_point(request):
    response_data = {
        'success': False,
        'plate_image_url': None,
        'plate_text': None,
        'plate_confidence': None,
        'plate_bbox': None,
        'text': "Doesn't have a driver license.",
    }
    
    try:
        s3_client = boto3.client('s3')
        rekognition_client = boto3.client('rekognition')
        
        # Read image data once
        image = request.FILES['image']
        image_data = image.read()
        file_extension = image.name.split('.')[-1].lower()
        content_type = image.content_type
        
        # First try face recognition
        temp_filename = f"temp/{uuid.uuid4()}.{file_extension}"
        with BytesIO(image_data) as image_buffer:
            s3_client.upload_fileobj(
                image_buffer,
                settings.AWS_STORAGE_BUCKET_NAME,
                temp_filename,
                ExtraArgs={'ContentType': content_type}
            )
        
        try:
            response = rekognition_client.search_faces_by_image(
                CollectionId=settings.AWS_REKOGNITION_COLLECTION_ID,
                Image={'S3Object': {
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Name': temp_filename
                }},
                MaxFaces=1,
                FaceMatchThreshold=90
            )
            
            if response['FaceMatches']:
                # Get driver details if face matched
                license_number = response['FaceMatches'][0]['Face']['ExternalImageId']
                try:
                    driver = Driver.objects.get(license_number=license_number)
                    driver_data = DriverSerializer(driver).data
                    response_data.update({
                        'success': True,
                        'text': "Driver recognized",
                        **driver_data  # Include all driver data in response
                    })
                    return Response(response_data)
                except Driver.DoesNotExist:
                    response_data['text'] = "Driver not found in database"
                    return Response(response_data, status=status.HTTP_404_NOT_FOUND)
                
        finally:
            s3_client.delete_object(
                Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                Key=temp_filename
            )

        # If no face match, proceed with plate recognition
        try:
            # Decode image for OpenCV
            image_array = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if cv_image is None or cv_image.size == 0:
                return Response({'error': 'Invalid image format'}, status=status.HTTP_400_BAD_REQUEST)

            # Initialize plate recognizer
            recognizer = EthiopianPlateRecognizer(debug_mode=True)
            plate_result = recognizer.process_image(cv_image)
            
            # Prepare plate image for upload
            plate_filename = f"plates/{uuid.uuid4()}.{file_extension}"
            
            # Use plate region if available, otherwise use whole image
            if plate_result.get('bbox'):
                x1, y1, x2, y2 = plate_result['bbox']
                plate_region = cv_image[y1:y2, x1:x2]
            else:
                plate_region = cv_image
            
            # Convert to bytes and upload
            success, encoded_image = cv2.imencode(f".{file_extension}", plate_region)
            if not success:
                raise ValueError("Failed to encode plate image")
            
            with BytesIO(encoded_image.tobytes()) as plate_buffer:
                s3_client.upload_fileobj(
                    plate_buffer,
                    settings.AWS_STORAGE_BUCKET_NAME,
                    plate_filename,
                    ExtraArgs={'ContentType': content_type}
                )
            
            # Update response data based on plate recognition results
            if plate_result['status'] == 'success':
                response_data.update({
                    'success': True,
                    'plate_image_url': f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{plate_filename}",
                    'plate_text': plate_result.get('plate_text'),
                    'plate_confidence': plate_result.get('confidence'),
                    'plate_bbox': plate_result.get('bbox'),
                    'text': "Vehicle plate recognized" if plate_result.get('plate_text') 
                           else "No readable plate detected"
                })
            else:
                response_data.update({
                    'plate_image_url': f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{plate_filename}",
                    'text': "Plate detected but could not be read",
                    'error': plate_result.get('error')
                })
            
            return Response(response_data)
            
        except Exception as e:
            # logger.error(f"Plate recognition error: {str(e)}")
            return Response(
                {
                    'error': 'Plate recognition failed',
                    'details': str(e),
                    'text': "Failed to process license plate"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    except ClientError as e:
        # logger.error(f"AWS client error: {str(e)}")
        return Response(
            {
                'error': 'Failed to process the image',
                'details': str(e),
                'text': "Image processing service unavailable"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except Exception as e:
        # logger.error(f"Checkpoint processing error: {str(e)}")
        return Response(
            {
                'error': 'Checkpoint processing failed',
                'details': str(e),
                'text': "Internal server error"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    

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
        return JsonResponse(
            {"status": "error", "error": "Internal server error"},
            status=500
        )