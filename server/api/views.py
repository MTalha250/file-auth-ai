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
import time
import os
import requests
import tempfile
from django.conf import settings
from scripts.universal_extractor import process_file_universal
from .models import (
    SubmittedFile,
    MLReferenceFile,
    AuditLog,
    CategorySchema,
    ProcessedFile
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
def submit_file1(request):
    """
    Upload and process submitted files for AI analysis.
    
    Handles document files (DOC, DOCX, PDF) using Cloudinary raw upload
    to prevent ZIP processing errors. Automatically detects file types
    and generates proper filenames with extensions.
    
    Expected fields:
    - category: File category for classification
    - file: The uploaded file
    - file_name: Name of the file (auto-corrected if needed)
    
    Returns analysis results including accuracy score and match status.
    """
    try:
        category = request.data.get('category')
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')

        if not all([category, file_obj, file_name]):
            return Response({"error": "Missing required fields."}, status=400)

        # Validate and clean filename
        if not file_name or file_name.strip() == '':
            return Response({"error": "File name cannot be empty."}, status=400)
        
        # Get proper filename from the uploaded file object
        original_filename = getattr(file_obj, 'name', '')
        
        # Determine file type - try multiple approaches
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        elif '.' in file_name:
            file_extension = file_name.lower().split('.')[-1]
        else:
            # Try to detect from content type
            content_type = getattr(file_obj, 'content_type', '')
            if 'pdf' in content_type.lower():
                file_extension = 'pdf'
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                file_extension = 'docx'
        
        # Create a proper filename
        if original_filename and '.' in original_filename:
            # Use the original filename if it's properly formatted
            final_filename = original_filename
        elif file_name and '.' in file_name and not file_name.lower() in ['pdf', 'doc', 'docx']:
            # Use provided filename if it's properly formatted
            final_filename = file_name
        else:
            # Generate a proper filename
            timestamp = int(time.time())
            if file_extension:
                final_filename = f"document_{timestamp}.{file_extension}"
            else:
                final_filename = f"document_{timestamp}.pdf"  # Default to PDF
                file_extension = 'pdf'
        
        # Generate unique identifier for file
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                # For document files, use 'raw' resource type to prevent ZIP processing
                public_id = f"submitted_files/{unique_id}_{final_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                # For images and other files, use auto detection
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="submitted_files",
                    public_id=f"{unique_id}_{final_filename}",
                    overwrite=True
                )
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=400)
            
        cloudinary_url = upload_result.get('secure_url')
        
        # For document files, ensure URL has proper file extension
        if file_extension in ['doc', 'docx', 'pdf']:
            # Always reconstruct the URL to ensure proper extension
            base_url = "https://res.cloudinary.com/dewqsghdi"
            version = upload_result.get('version', '')
            if version:
                cloudinary_url = f"{base_url}/raw/upload/v{version}/submitted_files/{unique_id}_{final_filename}"
            else:
                cloudinary_url = f"{base_url}/raw/upload/submitted_files/{unique_id}_{final_filename}"
        
        if not cloudinary_url:
            return Response({
                "error": "Failed to get upload URL from Cloudinary"
            }, status=500)

        submitted_file = SubmittedFile.objects.create(
            category=category,
            file=cloudinary_url,
            file_name=final_filename,
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
                "file_name": final_filename,
                "original_category": category,
                "final_category": final_category,
                "accuracy_score": score,
                "match": matched
            },
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT')
        )

        return Response({
            "file_name": final_filename,
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
    """
    Upload ML reference files for training and categorization.
    
    Restricted to admin users only. Handles document files (DOC, DOCX, PDF)
    using Cloudinary raw upload to prevent ZIP processing errors.
    Creates or updates category schemas as needed.
    
    Expected fields:
    - ml_reference_id: Unique identifier (auto-generated if not provided)
    - category: Category name for the reference
    - description: Description of the reference file
    - reasoning_notes: Explanation of why this file is a good reference
    - metadata: Additional JSON metadata (optional)
    - file: The uploaded reference file
    - file_name: Name of the file (auto-corrected if needed)
    
    Returns success message with the reference ID.
    """
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

        # Validate and clean filename
        if not file_name or file_name.strip() == '':
            return Response({"error": "File name cannot be empty."}, status=400)
        
        # Get proper filename from the uploaded file object
        original_filename = getattr(file_obj, 'name', '')
        
        # Determine file type - try multiple approaches
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        elif '.' in file_name:
            file_extension = file_name.lower().split('.')[-1]
        else:
            # Try to detect from content type
            content_type = getattr(file_obj, 'content_type', '')
            if 'pdf' in content_type.lower():
                file_extension = 'pdf'
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                file_extension = 'docx'
        
        # Create a proper filename
        if original_filename and '.' in original_filename:
            # Use the original filename if it's properly formatted
            final_filename = original_filename
        elif file_name and '.' in file_name and not file_name.lower() in ['pdf', 'doc', 'docx']:
            # Use provided filename if it's properly formatted
            final_filename = file_name
        else:
            # Generate a proper filename
            timestamp = int(time.time())
            if file_extension:
                final_filename = f"reference_{timestamp}.{file_extension}"
            else:
                final_filename = f"reference_{timestamp}.pdf"  # Default to PDF
                file_extension = 'pdf'
        
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                # For document files, use 'raw' resource type to prevent ZIP processing
                public_id = f"ml_references/{ml_reference_id}_{final_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                # For images and other files, use auto detection
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="ml_references",
                    public_id=f"{ml_reference_id}_{final_filename}",
                    overwrite=True
                )
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=400)
            
        cloudinary_url = upload_result.get('secure_url')
        
        # For document files, ensure URL has proper file extension
        if file_extension in ['doc', 'docx', 'pdf']:
            # Always reconstruct the URL to ensure proper extension
            base_url = "https://res.cloudinary.com/dewqsghdi"
            version = upload_result.get('version', '')
            if version:
                cloudinary_url = f"{base_url}/raw/upload/v{version}/ml_references/{ml_reference_id}_{final_filename}"
            else:
                cloudinary_url = f"{base_url}/raw/upload/ml_references/{ml_reference_id}_{final_filename}"
        
        if not cloudinary_url:
            return Response({
                "error": "Failed to get upload URL from Cloudinary"
            }, status=500)

        category_obj, _ = CategorySchema.objects.get_or_create(
            category_name=category_name,
            defaults={"description": description or "", "expected_fields": []}
        )

        reference = MLReferenceFile.objects.create(
            ml_reference_id=ml_reference_id,
            file=cloudinary_url,
            file_name=final_filename,
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def submit_file(request):
    """
    Upload and process submitted files for AI analysis and universal text extraction.
    
    Handles document files (DOC, DOCX, PDF, images) using Cloudinary raw upload.
    Processes files with the universal extractor and stores results.
    
    Expected fields:
    - category: File category for classification
    - file: The uploaded file
    - file_name: Name of the file (auto-corrected if needed)
    
    Returns analysis results including accuracy score, match status, and extracted text.
    """
    try:
        category = request.data.get('category')
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')

        if not all([category, file_obj, file_name]):
            return Response({"error": "Missing required fields."}, status=400)

        # Validate and clean filename
        if not file_name or file_name.strip() == '':
            return Response({"error": "File name cannot be empty."}, status=400)
        
        # Get proper filename from the uploaded file object
        original_filename = getattr(file_obj, 'name', '')
        
        # Determine file type
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        elif '.' in file_name:
            file_extension = file_name.lower().split('.')[-1]
        else:
            content_type = getattr(file_obj, 'content_type', '')
            if 'pdf' in content_type.lower():
                file_extension = 'pdf'
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                file_extension = 'docx'
            elif 'image' in content_type.lower():
                file_extension = 'jpg'
        
        # Create a proper filename
        if original_filename and '.' in original_filename:
            final_filename = original_filename
        elif file_name and '.' in file_name and not file_name.lower() in ['pdf', 'doc', 'docx']:
            final_filename = file_name
        else:
            timestamp = int(time.time())
            if file_extension:
                final_filename = f"document_{timestamp}.{file_extension}"
            else:
                final_filename = f"document_{timestamp}.pdf"
                file_extension = 'pdf'
        
        # Generate unique identifier for file
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Upload to Cloudinary
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                public_id = f"submitted_files/{unique_id}_{final_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="submitted_files",
                    public_id=f"{unique_id}_{final_filename}",
                    overwrite=True
                )
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=400)
            
        cloudinary_url = upload_result.get('secure_url')
        
        # Reconstruct URL for documents
        if file_extension in ['doc', 'docx', 'pdf']:
            base_url = "https://res.cloudinary.com/dewqsghdi"
            version = upload_result.get('version', '')
            if version:
                cloudinary_url = f"{base_url}/raw/upload/v{version}/submitted_files/{unique_id}_{final_filename}"
            else:
                cloudinary_url = f"{base_url}/raw/upload/submitted_files/{unique_id}_{final_filename}"
        
        if not cloudinary_url:
            return Response({
                "error": "Failed to get upload URL from Cloudinary"
            }, status=500)

        # Create SubmittedFile instance
        submitted_file = SubmittedFile.objects.create(
            category=category,
            file=cloudinary_url,
            file_name=final_filename,
            uploaded_by=request.user,
            status='processing'
        )

        # Run existing AI analysis
        score, matched, final_category, extracted = run_ai_analysis(submitted_file)
        submitted_file.accuracy_score = score
        submitted_file.match = matched
        submitted_file.final_category = final_category
        submitted_file.extracted_fields = extracted
        submitted_file.status = 'completed'
        submitted_file.processed_at = timezone.now()
        submitted_file.save()

        # Process with UniversalTextExtractor
        try:
            # Download file to temporary location
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, final_filename)
            response = requests.get(cloudinary_url, timeout=120)
            response.raise_for_status()
            
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(response.content)

            # Process file with universal extractor
            extractor_results = process_file_universal(
                file_input=temp_file_path,
                output_dir=os.path.join(settings.BASE_DIR, 'extracted_texts'),
                use_ocr=True,
                enable_gpu=True
            )

            # Clean up temporary file
            try:
                os.remove(temp_file_path)
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary files: {cleanup_error}")

            # Create ProcessedFile instance
            processed_file = ProcessedFile.objects.create(
                submitted_file=submitted_file,
                extracted_text=extractor_results.get('text', '') + extractor_results.get('ocr_text', ''),
                extracted_metadata={
                    'file_type': extractor_results.get('file_type', 'unknown'),
                    'method': extractor_results.get('method', 'unknown'),
                    'pages': extractor_results.get('pages', 0),
                    'images_found': extractor_results.get('images_found', 0),
                    'images_processed': extractor_results.get('images_processed', 0),
                    'confidence': extractor_results.get('confidence', 0),
                    'output_file': extractor_results.get('output_file', ''),
                },
                status='completed' if 'error' not in extractor_results else 'failed',
                error_message=extractor_results.get('error', '')
            )

        except Exception as extractor_error:
            processed_file = ProcessedFile.objects.create(
                submitted_file=submitted_file,
                extracted_text='',
                extracted_metadata={},
                status='failed',
                error_message=str(extractor_error)
            )
            print(f"Extractor error: {extractor_error}")

        # Create audit log
        AuditLog.objects.create(
            action='analysis',
            user=request.user,
            submitted_file=submitted_file,
            details={
                "file_name": final_filename,
                "original_category": category,
                "final_category": final_category,
                "accuracy_score": score,
                "match": matched,
                "extractor_status": processed_file.status,
                "extractor_error": processed_file.error_message
            },
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT')
        )

        return Response({
            "file_name": final_filename,
            "original_category": category,
            "final_category": final_category,
            "accuracy_score": score,
            "match": matched,
            "extracted_text": processed_file.extracted_text,
            "extracted_metadata": processed_file.extracted_metadata
        }, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
