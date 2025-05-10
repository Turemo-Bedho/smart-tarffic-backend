# from rest_framework import serializers

# class GoogleLoginSerializer(serializers.Serializer):
#     token = serializers.CharField()

from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from . import models

class UserCreateSerializer(BaseUserCreateSerializer):
    class Meta(BaseUserCreateSerializer.Meta):
        model = models.User
        fields = ['email', 'password', 'first_name', 'username']