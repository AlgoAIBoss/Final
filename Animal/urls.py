from django.urls import path
from .views import *

urlpatterns = [
    path('', MyFormView.as_view(), name='upload'),
]
