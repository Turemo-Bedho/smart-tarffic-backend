# import requests
# from .serializers import GoogleLoginSerializer
# from django.shortcuts import render
# from rest_framework.mixins import CreateModelMixin
# from rest_framework.viewsets import GenericViewSet
# from rest_framework.response import Response
# from rest_framework import status
# from allauth.socialaccount.models import SocialAccount
# from django.contrib.auth import get_user_model
# from rest_framework_simplejwt.tokens import RefreshToken

# # Create your views here.

# class GoogleLoginView(GenericViewSet, CreateModelMixin):
#     serializer_class = GoogleLoginSerializer

#     def create(self, request, *args, **kwargs):
#         google_token = request.data.get('token')
#         if google_token:
#             try:
#                 # Validate the Google token and extract user info
#                 user_info = self.get_google_user_info(google_token)

                
#                 # Check if the user exists or create a new one
#                 User = get_user_model()
#                 user, created = User.objects.get_or_create(
#                     username=user_info['email'],
#                     defaults={'email': user_info['email']}
#                 )
                
#                 # Link Google account to the user
#                 if created:
#                     SocialAccount.objects.create(
#                         user=user,
#                         provider='google',
#                         uid=user_info['id'],
#                         extra_data=user_info
#                     )
                
#                 # Create and return JWT token after user is authenticated/created
#                 refresh = RefreshToken.for_user(user)
#                 return Response({
#                     'refresh': str(refresh),
#                     'access': str(refresh.access_token),
#                 }, status=status.HTTP_201_CREATED)

#             except Exception as e:
#                 print(e)
#                 return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

#         return Response({"error": "Google token is required."}, status=status.HTTP_400_BAD_REQUEST)
    
#     def get_google_user_info(self, token):
#         url = f'https://www.googleapis.com/oauth2/v3/userinfo?access_token={token}'
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return None