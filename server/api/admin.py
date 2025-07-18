from django.contrib import admin
from django.contrib.auth.models import User, Group
from unfold.admin import ModelAdmin
from .models import UserProfile, CategorySchema, MLReferenceFile, SubmittedFile, AuditLog

# Unregister the default User and Group admin
admin.site.unregister(User)
admin.site.unregister(Group)

# User admin with Unfold
@admin.register(User)
class UserAdmin(ModelAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff']
    list_filter = ['is_staff', 'is_active']
    search_fields = ['username', 'email']
    list_display_links = ['username']

# Group admin with Unfold
@admin.register(Group)
class GroupAdmin(ModelAdmin):
    list_display = ['name']
    search_fields = ['name']
    list_display_links = ['name']

# UserProfile admin
@admin.register(UserProfile)
class UserProfileAdmin(ModelAdmin):
    list_display = ['user', 'role', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username']
    readonly_fields = ['created_at']
    fieldsets = (
        (None, {
            'fields': ('user', 'role')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
    list_display_links = ['user']

# CategorySchema admin
@admin.register(CategorySchema)
class CategorySchemaAdmin(ModelAdmin):
    list_display = ('category_name', 'description_short', 'created_at')
    search_fields = ('category_name',)
    list_filter = ('created_at',)
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('category_name', 'description', 'expected_fields')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    list_display_links = ['category_name']

    def description_short(self, obj):
        """Truncate description for list display."""
        return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
    description_short.short_description = 'Description'

# MLReferenceFile admin
@admin.register(MLReferenceFile)
class MLReferenceFileAdmin(ModelAdmin):
    list_display = ('ml_reference_id', 'file_name', 'category', 'uploaded_at')
    search_fields = ('ml_reference_id', 'file_name')
    list_filter = ('category', 'uploaded_at')
    readonly_fields = ('uploaded_at',)
    fieldsets = (
        (None, {
            'fields': (
                'ml_reference_id',
                'file_name',
                'file',
                'category',
                'description',
                'reasoning_notes',
                'metadata',
                'uploaded_by'
            )
        }),
        ('Upload Info', {
            'fields': ('uploaded_at',)
        }),
    )
    list_display_links = ['ml_reference_id']
    raw_id_fields = ['uploaded_by']

# SubmittedFile admin
@admin.register(SubmittedFile)
class SubmittedFileAdmin(ModelAdmin):
    list_display = (
        'case_id', 'file_name', 'category',
        'final_category', 'accuracy_score', 'match', 'status', 'uploaded_at'
    )
    search_fields = ('case_id', 'file_name')
    list_filter = ('category', 'final_category', 'match', 'status', 'uploaded_at')
    readonly_fields = ('uploaded_at', 'processed_at')
    fieldsets = (
        (None, {
            'fields': (
                'case_id',
                'file_name',
                'file',
                'category',
                'final_category',
                'accuracy_score',
                'match',
                'extracted_fields',
                'uploaded_by',
                'status',
                'error_message'
            )
        }),
        ('Submission Info', {
            'fields': ('uploaded_at', 'processed_at')
        }),
    )
    list_display_links = ['case_id']
    raw_id_fields = ['uploaded_by']

# AuditLog admin
@admin.register(AuditLog)
class AuditLogAdmin(ModelAdmin):
    list_display = (
        'timestamp', 'user', 'action', 'submitted_file', 'ml_reference_file'
    )
    search_fields = ('user__username', 'action')
    list_filter = ('action', 'timestamp')
    readonly_fields = ('timestamp',)
    fieldsets = (
        (None, {
            'fields': (
                'user',
                'action',
                'submitted_file',
                'ml_reference_file',
                'details',
                'ip_address',
                'user_agent'
            )
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        }),
    )
    list_display_links = ['timestamp']
    raw_id_fields = ['user', 'submitted_file', 'ml_reference_file']