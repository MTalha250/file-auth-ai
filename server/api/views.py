from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import generics, serializers, status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from django.utils import timezone
from cloudinary.uploader import upload as cloudinary_upload
import uuid
import random

from .models import (
    SubmittedFile,
    MLReferenceFile,
    AuditLog,
    CategorySchema
)
from .serializers import CategorySchemaSerializer


# AUTHENTICATION VIEWS
@api_view(['GET'])
@permission_classes([AllowAny])
def index(request):
    return Response({
        'message': 'Welcome to the FileAuthAI API',
        'status': 'success',
        'version': '1.0',
        'endpoints': {
            'auth': {
                'login': '/auth/login/',
                'refresh': '/auth/refresh/',
                'register': '/auth/register/',
                'profile': '/auth/profile/',
            },
            'api': {
                'submit_file': '/api/v1/submit-file/',
                'upload_reference': '/api/v1/upload-ml-reference/',
            }
        }
    })


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm', 'first_name', 'last_name')

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'date_joined', 'is_active')
        read_only_fields = ('id', 'username', 'date_joined', 'is_active')


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'message': 'User registered successfully',
            'user': UserProfileSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


# AI LOGIC PLACEHOLDER 
def run_ai_analysis(submitted_file: SubmittedFile):
    score = round(random.uniform(50.0, 100.0), 2)
    matched = 'Y' if score >= 80 else 'N'
    final_category = submitted_file.category if matched == 'Y' else 'unknown_category'

    extracted_fields = {
        "passenger_name": "John Doe",
        "flight_number": "XY123",
        "flight_date": "2025-07-20"
    } if "flight" in submitted_file.category else {}

    return score, matched, final_category, extracted_fields


# FILE SUBMISSION 
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def submit_file(request):
    try:
        case_id = request.data.get('case_id')
        category = request.data.get('category')
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')

        if not all([case_id, category, file_obj, file_name]):
            return Response({"error": "Missing required fields."}, status=400)

        upload_result = cloudinary_upload(file_obj)
        cloudinary_url = upload_result.get('secure_url')

        submitted_file = SubmittedFile.objects.create(
            case_id=case_id,
            category=category,
            file=cloudinary_url,
            file_name=file_name,
            uploaded_by=request.user,
            status='processing'
        )

        score, matched, final_category, extracted = run_ai_analysis(submitted_file)
        submitted_file.accuracy_score = score
        submitted_file.match = matched
        submitted_file.final_category = final_category
        submitted_file.extracted_fields = extracted
        submitted_file.status = 'completed'
        submitted_file.processed_at = timezone.now()
        submitted_file.save()

        AuditLog.objects.create(
            action='analysis',
            user=request.user,
            submitted_file=submitted_file,
            details={
                "original_category": category,
                "final_category": final_category,
                "accuracy_score": score,
                "match": matched
            },
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT')
        )

        return Response({
            "file_name": file_name,
            "case_id": case_id,
            "original_category": category,
            "final_category": final_category,
            "accuracy_score": score,
            "match": matched
        }, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)


#  ML REFERENCE UPLOAD 
@api_view(['GET'])
@permission_classes([AllowAny])
def category_list(request):
    categories = CategorySchema.objects.all()
    serializer = CategorySchemaSerializer(categories, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_ml_reference(request):
    try:
        if request.user.userprofile.role != 'admin':
            return Response({"error": "Unauthorized"}, status=403)

        ml_reference_id = request.data.get('ml_reference_id') or str(uuid.uuid4())
        category_name = request.data.get('category')
        description = request.data.get('description')
        reasoning_notes = request.data.get('reasoning_notes')
        metadata = request.data.get('metadata')  # Should be JSON string or dict
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')

        if not all([category_name, description, reasoning_notes, file_obj, file_name]):
            return Response({"error": "Missing required fields."}, status=400)

        upload_result = cloudinary_upload(file_obj)
        cloudinary_url = upload_result.get('secure_url')

        category_obj, _ = CategorySchema.objects.get_or_create(
            category_name=category_name,
            defaults={"description": description or "", "expected_fields": []}
        )

        reference = MLReferenceFile.objects.create(
            ml_reference_id=ml_reference_id,
            file=cloudinary_url,
            file_name=file_name,
            category=category_obj,
            description=description,
            reasoning_notes=reasoning_notes,
            metadata=metadata or {},
            uploaded_by=request.user
        )

        AuditLog.objects.create(
            action='upload',
            user=request.user,
            ml_reference_file=reference,
            details={"category": category_name},
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT')
        )

        return Response({"message": "ML reference uploaded", "id": ml_reference_id}, status=201)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
