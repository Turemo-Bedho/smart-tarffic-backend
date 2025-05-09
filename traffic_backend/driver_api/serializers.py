from rest_framework import serializers
from .models import Driver, Address, Officer
from django.contrib.auth import get_user_model
from .utils import get_face_embedding

User = get_user_model()

class OfficerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Officer
        fields = ['badge_number']
    
    def create(self, validated_data):
        validated_data['user_id'] = self.context['request'].user.id
        return super().create(validated_data)


class AddressSerializer(serializers.ModelSerializer):
    driver = serializers.PrimaryKeyRelatedField(queryset=Driver.objects.all())

    class Meta:
        model = Address
        fields = [
            'id', 'driver', 'region', 'woreda', 'house_number',
            'street', 'city', 'postal_code', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class DriverSerializer(serializers.ModelSerializer):
    addresses = AddressSerializer(many=True, read_only=True)

    class Meta:
        model = Driver
        fields = [
            'id', 'license_number', 'first_name', 'middle_name', 'last_name',
            'date_of_birth', 'sex', 'phone_number', 'nationality',
            'license_issue_date', 'blood_type', 'profile_image', 'created_at', 'updated_at', 'addresses'
        ]
        read_only_fields = ['created_at', 'updated_at']