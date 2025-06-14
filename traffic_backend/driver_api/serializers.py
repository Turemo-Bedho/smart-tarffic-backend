from rest_framework import serializers
from .models import Driver, Address, Officer, ViolationType, Vehicle, Violation
from django.contrib.auth import get_user_model


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


class ViolationTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ViolationType
        fields = ['id', 'name', 'description', 'fine_amount', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']

class VehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vehicle
        fields = [
            'id', 'license_plate', 'make', 'model', 'year',
            'color', 'vin', 'registration_date', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

class ViolationSerializer(serializers.ModelSerializer):
    driver = serializers.PrimaryKeyRelatedField(queryset=Driver.objects.all(), write_only=True)
    violation_types = serializers.PrimaryKeyRelatedField(  # Changed field name (plural)
        queryset=ViolationType.objects.all(),
        many=True,  # Critical change for M2M
        write_only=True
    )
    license_plate = serializers.CharField(max_length=20, write_only=True)
    
    # Read-only representations
    vehicle_detail = VehicleSerializer(source='vehicle', read_only=True)
    driver_detail = DriverSerializer(source='driver', read_only=True)
    violation_types_detail = ViolationTypeSerializer(  # Changed field name (plural)
        source='violation_type', 
        many=True,  # Reflects M2M relationship
        read_only=True
    )
    officer_detail = OfficerSerializer(source='issued_by_officer', read_only=True)

    class Meta:
        model = Violation
        fields = [
            'id',
            # Writeable fields
            'driver', 'violation_types', 'license_plate', 'location',  # Changed field name
            # Read-only detailed representations
            'vehicle_detail', 'driver_detail', 'violation_types_detail', 'officer_detail',  # Changed
            # Automatic fields
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

    def validate(self, data):
        # License plate validation remains the same
        try:
            data['vehicle'] = Vehicle.objects.get(license_plate=data['license_plate'])
        except Vehicle.DoesNotExist:
            raise serializers.ValidationError({
                'license_plate': 'Vehicle with this license plate does not exist'
            })
        return data

    def create(self, validated_data):
        # Extract M2M data before creation
        violation_types = validated_data.pop('violation_types', [])
        validated_data['issued_by_officer_id'] = self.context['user_id']
        validated_data.pop('license_plate', None)
        
        # Create violation instance
        violation = super().create(validated_data)
        
        # Set M2M relationship
        violation.violation_type.set(violation_types)  # Changed to use .set()
        
        return violation

