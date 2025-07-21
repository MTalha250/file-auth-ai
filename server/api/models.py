from django.contrib.auth.models import User
from django.db import models
from cloudinary.models import CloudinaryField


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('user', 'Portal User'), 
        ('admin', 'Client Admin'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"
    
    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"


class CategorySchema(models.Model):
    category_name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    expected_fields = models.JSONField(
        default=list,
        help_text="List of fields to extract via OCR, e.g., ['passenger_name', 'flight_number']"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.category_name
    
    class Meta:
        verbose_name = "Category Schema"
        verbose_name_plural = "Category Schemas"
        ordering = ['-created_at']

class MLReferenceFile(models.Model):
    ml_reference_id = models.CharField(max_length=50, unique=True)
    file = CloudinaryField('reference_file')
    file_name = models.CharField(max_length=255)
    category = models.ForeignKey(
        CategorySchema,
        on_delete=models.CASCADE,
        related_name='ml_reference_files'
    )
    description = models.TextField()
    reasoning_notes = models.TextField(
        help_text="Why this file is a good reference for this category"
    )
    metadata = models.JSONField(
        default=dict,
        help_text="Additional metadata like document_type, region, issue_year"
    )
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        limit_choices_to={'userprofile__role': 'admin'}
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.ml_reference_id} - {self.category.category_name}"
    
    class Meta:
        verbose_name = "ML Reference File"
        verbose_name_plural = "ML Reference Files"
        ordering = ['-uploaded_at']


class SubmittedFile(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending Analysis'),
        ('processing', 'Being Processed'),
        ('completed', 'Analysis Complete'),
        ('failed', 'Processing Failed'),
    ]
    
    MATCH_CHOICES = [
        ('Y', 'Yes'),
        ('N', 'No'),
    ]
    
    case_id = models.CharField(max_length=50)
    file = CloudinaryField('submitted_file')
    file_name = models.CharField(max_length=255)
    category = models.CharField(
        max_length=100,
        help_text="Original category submitted by user"
    )
    final_category = models.CharField(
        max_length=100, 
        blank=True,
        help_text="Final category after AI analysis (may differ from original)"
    )
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        limit_choices_to={'userprofile__role': 'user'}
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # AI Analysis Results
    accuracy_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Accuracy score from 0-100"
    )
    match = models.CharField(
        max_length=1, 
        choices=MATCH_CHOICES, 
        blank=True,
        help_text="Y if accuracy >= threshold, N if below"
    )
    extracted_fields = models.JSONField(
        default=dict, 
        blank=True,
        help_text="OCR extracted data like names, dates, numbers"
    )
    
    # Processing Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    processed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.case_id} - {self.file_name}"
    
    class Meta:
        verbose_name = "Submitted File"
        verbose_name_plural = "Submitted Files"
        ordering = ['-uploaded_at']


class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('upload', 'File Upload'),
        ('analysis', 'AI Analysis'),
        ('re-categorization', 'Category Change'),
        ('login', 'User Login'),
        ('admin_action', 'Admin Action'),
    ]
    
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # File References
    submitted_file = models.ForeignKey(
        SubmittedFile, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    ml_reference_file = models.ForeignKey(
        MLReferenceFile, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    # Audit Details
    details = models.JSONField(
        default=dict,
        help_text="Additional details like from_category, to_category, error_info"
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.action} by {self.user.username} at {self.timestamp}"
    
    class Meta:
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"
        ordering = ['-timestamp']


# Signal to create UserProfile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'userprofile'):
        instance.userprofile.save()
    else:
        UserProfile.objects.create(user=instance)
