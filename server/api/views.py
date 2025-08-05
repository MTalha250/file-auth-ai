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
import re
from django.db import transaction
from rest_framework.exceptions import ValidationError
from datetime import datetime
import logging
from scripts.universal_extractor import UniversalTextExtractor
from rest_framework.parsers import JSONParser
logger = logging.getLogger(__name__)
from .serializers import (
    SubmittedFileSerializer, 
    
    
)
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
#@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def submit_file(request):
    """
    Accepts a Cloudinary file URL from the frontend, processes it with UniversalTextExtractor,
    and saves results to SubmittedFile and ProcessedFile models.
    
    Expected JSON payload:
    - file: Cloudinary file URL (required)
    - category: File category (optional, defaults to 'unknown')
    - metadata: Additional JSON metadata (optional)
    
    Returns:
    - SubmittedFile data with processing results, including extracted text and metadata
    """
    try:
        # Extract and validate input
        file_url = request.data.get('file')
        category_name = request.data.get('category', 'unknown')
        metadata = request.data.get('metadata', {})

        if not file_url:
            return Response({"error": "No file URL provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate URL format (basic check for Cloudinary or similar URL)
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', file_url):
            return Response({"error": "Invalid file URL format"}, status=status.HTTP_400_BAD_REQUEST)

        # Extract file name from URL
        file_name = file_url.split('/')[-1].split('?')[0]
        if not file_name:
            return Response({"error": "Could not extract file name from URL"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate category
        category_obj = CategorySchema.objects.filter(category_name=category_name).first()
        if not category_obj:
            return Response({"error": f"Invalid category: {category_name}"}, status=status.HTTP_400_BAD_REQUEST)

        # Create SubmittedFile record
        with transaction.atomic():
            submitted_file = SubmittedFile.objects.create(
                file=file_url,
                file_name=file_name,
                category=category_name,  # Store as CharField per your model
                uploaded_by=request.user,
                status='processing',
                extracted_fields={}
            )

            # Create AuditLog for upload
            AuditLog.objects.create(
                action='upload',
                user=request.user,
                submitted_file=submitted_file,
                details={
                    'category': category_name,
                    'file_url': file_url,
                    'file_name': file_name,
                    'metadata': metadata
                },
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT')
            )

            try:
                # Initialize UniversalTextExtractor
                extractor = UniversalTextExtractor(
                    output_dir="extracted_texts",
                    use_ocr=True,
                    enable_gpu=True
                )

                # Process the file URL
                extraction_results = extractor.process_file(file_url)
                if 'error' in extraction_results:
                    raise ValueError(f"Extraction failed: {extraction_results['error']}")

                # Run AI analysis
                accuracy_score, match, final_category, extracted_fields = run_ai_analysis(submitted_file)

                # Update SubmittedFile with processing results
                submitted_file.status = 'completed'
                submitted_file.processed_at = timezone.now()
                submitted_file.accuracy_score = accuracy_score
                submitted_file.match = match
                submitted_file.final_category = final_category
                submitted_file.extracted_fields = extracted_fields
                submitted_file.save()

                # Create ProcessedFile record
                processed_file = ProcessedFile.objects.create(
                    submitted_file=submitted_file,
                    extracted_text=extraction_results.get('text', ''),
                    extracted_metadata={
                        'pages': extraction_results.get('pages', 0),
                        'images_found': extraction_results.get('images_found', 0),
                        'images_processed': extraction_results.get('images_processed', 0),
                        'method': extraction_results.get('method', 'unknown'),
                        'file_type': extraction_results.get('file_type', 'unknown')
                    },
                    status='completed'
                )

                # Create AuditLog for analysis
                AuditLog.objects.create(
                    action='analysis',
                    user=request.user,
                    submitted_file=submitted_file,
                    details={
                        'final_category': final_category,
                        'accuracy_score': accuracy_score,
                        'match': match,
                        'extracted_fields': list(extracted_fields.keys()),
                        'extractor_metadata': extraction_results.get('extracted_metadata', {})
                    },
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )

                # Serialize and return response
                serializer = SubmittedFileSerializer(submitted_file)
                return Response(serializer.data, status=status.HTTP_201_CREATED)

            except Exception as processing_error:
                logger.error(f"File processing failed for {file_url}: {str(processing_error)}", exc_info=True)
                submitted_file.status = 'failed'
                submitted_file.error_message = str(processing_error)
                submitted_file.processed_at = timezone.now()
                submitted_file.save()

                ProcessedFile.objects.create(
                    submitted_file=submitted_file,
                    status='failed',
                    error_message=str(processing_error)
                )

                AuditLog.objects.create(
                    action='file_processing_failed',
                    user=request.user,
                    submitted_file=submitted_file,
                    details={
                        'error': str(processing_error),
                        'category': category_name,
                        'file_name': file_name,
                        'file_url': file_url
                    },
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )

                return Response(
                    {"error": f"File processing failed: {str(processing_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.error(f"Unexpected error in file submission: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred during file submission"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )