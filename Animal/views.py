# articles/views.py
from .forms import ImageForm
from django.shortcuts import render
from .Model import *
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
import os
# print(os. getcwd())


class MyFormView(LoginRequiredMixin, View):
    form_class = ImageForm
    initial = {'key': 'value'}
    template_name = 'upload.html'

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        context = {}
        form = self.form_class(request.FILES)
        if bool(request.FILES.get('photo', False)) == True:
            photo = request.FILES['photo']
            context['photo'] = reader(photo)
            context['answer'] = predict(photo)
            return render(request, 'result.html', context)

        return render(request, self.template_name, {'form': form})
