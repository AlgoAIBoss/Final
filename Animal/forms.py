from django import forms


class ImageForm(forms.Form):
    photo = forms.ImageField(required=True)
