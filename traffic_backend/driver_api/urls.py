
# from django.urls import path
# from .views import DriverView, FaceRecognitionView

# urlpatterns = [
#     # URL for retrieving driver information
#     path('driver/<str:license_number>/', DriverView.as_view(), name='driver-detail'),

#     # URL for face recognition
#     path('recognize-face/', FaceRecognitionView.as_view(), name='recognize-face'),
# ]

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DriverView, OfficerView, AddressView

router = DefaultRouter()
router.register('drivers', DriverView, basename='driver')
router.register('officers', OfficerView, basename='officer')
router.register('addresses', AddressView, basename='address')

urlpatterns = router.urls
