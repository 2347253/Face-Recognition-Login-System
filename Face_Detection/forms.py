from django import forms
from .models import UserProfile

class RegistrationForm(forms.ModelForm):  # Fix typo in class name
    class Meta:
        model = UserProfile
        fields = [
            'face_id',
            'name',
            'address',
            'job',
            'phone',
            'email',
            'bio',
            'image',
        ]

    image = forms.ImageField(required=False)  # Ensure image field is correctly handled
