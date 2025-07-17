from django.contrib import admin
from django.contrib.auth.models import User, Group
from unfold.admin import ModelAdmin

# Unregister the default User and Group admin first
admin.site.unregister(User)
admin.site.unregister(Group)

# Simple User admin with Unfold
@admin.register(User)
class UserAdmin(ModelAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff']
    list_filter = ['is_staff', 'is_active']
    search_fields = ['username', 'email']

# Simple Group admin with Unfold
@admin.register(Group)
class GroupAdmin(ModelAdmin):
    list_display = ['name']
    search_fields = ['name']
