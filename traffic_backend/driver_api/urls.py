from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DriverView, OfficerView, AddressView, check_point, ViolationView, ViolationTypeView, VehicleView

router = DefaultRouter()
router.register('drivers', DriverView, basename='driver')
router.register('officers', OfficerView, basename='officer')
router.register('addresses', AddressView, basename='address')
router.register('violations', ViolationView, basename='violation')
router.register('violation-types', ViolationTypeView, basename='violation-type')
router.register('vehicles', VehicleView, basename='vehicles')

urlpatterns = router.urls + [path('check-point/', check_point, name='check_point')]