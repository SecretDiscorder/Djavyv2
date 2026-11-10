from django import forms
from .models import QuizResult

class UserInfoForm(forms.ModelForm):
    class Meta:
        model = QuizResult
        fields = ['name', 'number']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'number': forms.TextInput(attrs={'class': 'form-control'}),
        }

