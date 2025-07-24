from django.urls import path
from . import views

app_name = 'api'

from .views import (
    index, RegisterView, UserProfileView,
    submit_file, upload_ml_reference, category_list
)

urlpatterns = [
    path('', index),
    path('v1/submit-file/', submit_file),
    path('v1/upload-ml-reference/', upload_ml_reference),
    path('v1/categories/', category_list),
]
