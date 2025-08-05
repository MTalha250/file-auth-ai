from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, CategorySchema, MLReferenceFile, SubmittedFile, AuditLog,ProcessedFile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']
class ProcessedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessedFile
        fields = ['extracted_text', 'extracted_metadata', 'processed_at', 'status', 'error_message']

class SubmittedFileSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)
    processed_file = ProcessedFileSerializer(read_only=True)

    class Meta:
        model = SubmittedFile
        fields = [
            'id', 'file', 'file_name', 'category', 'final_category',
            'uploaded_by', 'uploaded_at', 'accuracy_score', 'match', 'extracted_fields',
            'status', 'processed_at', 'error_message', 'processed_file'
        ]        

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    class Meta:
        model = UserProfile
        fields = ['id', 'user', 'role', 'created_at']

class CategorySchemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = CategorySchema
        fields = ['id', 'category_name', 'description', 'expected_fields', 'created_at', 'updated_at']

class MLReferenceFileSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)
    category = serializers.SlugRelatedField(
        slug_field='category_name',
        queryset=CategorySchema.objects.all()
    )
    class Meta:
        model = MLReferenceFile
        fields = [
            'id', 'ml_reference_id', 'file', 'file_name', 'category', 'description',
            'reasoning_notes', 'metadata', 'uploaded_by', 'uploaded_at'
        ]


class AuditLogSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    submitted_file = SubmittedFileSerializer(read_only=True)
    ml_reference_file = MLReferenceFileSerializer(read_only=True)
    class Meta:
        model = AuditLog
        fields = [
            'id', 'action', 'user', 'timestamp', 'submitted_file', 'ml_reference_file',
            'details', 'ip_address', 'user_agent'
        ] 