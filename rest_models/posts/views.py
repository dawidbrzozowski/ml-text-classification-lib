from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from . import models
from . import serializers


class ListCreatePost(generics.ListCreateAPIView):
    queryset = models.Post.objects.all()
    serializer_class = serializers.PostSerializer
