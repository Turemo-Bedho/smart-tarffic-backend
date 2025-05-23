from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DriverView, OfficerView, AddressView

router = DefaultRouter()
router.register('drivers', DriverView, basename='driver')
router.register('officers', OfficerView, basename='officer')
router.register('addresses', AddressView, basename='address')

urlpatterns = router.urls
