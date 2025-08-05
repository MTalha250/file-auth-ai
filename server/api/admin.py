from django.contrib import admin
from django.contrib.auth.models import User, Group
from django.utils.html import format_html
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
    list_display = ('ml_reference_id', 'file_name', 'category_name', 'uploaded_at')
    search_fields = ('ml_reference_id', 'file_name', 'category__category_name')
    list_filter = ('category__category_name', 'uploaded_at')
    readonly_fields = ('uploaded_at',)
    autocomplete_fields = ['category', 'uploaded_by']
    fieldsets = (
        (None, {
            'fields': (
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

    def category_name(self, obj):
        """Display the category name in list view."""
        return obj.category.category_name
    category_name.short_description = 'Category'

# SubmittedFile admin
@admin.register(SubmittedFile)
class SubmittedFileAdmin(ModelAdmin):
    list_display = (
        'file_name', 'category', 'file_link',
        'final_category', 'accuracy_score', 'match', 'status', 'uploaded_at'
    )
    search_fields = ('file_name',)
    list_filter = ('category', 'final_category', 'match', 'status', 'uploaded_at')
    readonly_fields = (
        'file_name', 'file', 'category', 'final_category',
        'accuracy_score', 'match', 'extracted_fields', 'uploaded_by',
        'status', 'error_message', 'uploaded_at', 'processed_at'
    )
    fieldsets = (
        (None, {
            'fields': (
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
    list_display_links = ['file_name']
    raw_id_fields = ['uploaded_by']
    
    def file_link(self, obj):
        """Display the file as a clickable link with proper URL."""
        if obj.file:
            file_url = str(obj.file)
            
            # If it's already a full URL, use it
            if file_url.startswith('http'):
                url = file_url
            else:
                # Construct proper Cloudinary URL
                if obj.file_name and '.' in obj.file_name:
                    extension = obj.file_name.split('.')[-1].lower()
                    if extension in ['doc', 'docx', 'pdf']:
                        url = f"https://res.cloudinary.com/dewqsghdi/raw/upload/{file_url}"
                    else:
                        url = f"https://res.cloudinary.com/dewqsghdi/image/upload/{file_url}"
                else:
                    url = f"https://res.cloudinary.com/dewqsghdi/raw/upload/{file_url}"
            
            return format_html('<a href="{}" target="_blank">{}</a>', url, obj.file_name or 'View File')
        return "No file"
    file_link.short_description = 'File'

    def has_add_permission(self, request):
        """Prevent admins from creating SubmittedFile records."""
        return False

    def has_change_permission(self, request, obj=None):
        """Prevent admins from editing SubmittedFile records."""
        return False

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