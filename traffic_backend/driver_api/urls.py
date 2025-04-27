
from django.urls import path
from .views import DriverView, FaceRecognitionView

urlpatterns = [
    # URL for retrieving driver information
    path('driver/<str:license_number>/', DriverView.as_view(), name='driver-detail'),

    # URL for face recognition
    path('recognize-face/', FaceRecognitionView.as_view(), name='recognize-face'),
]