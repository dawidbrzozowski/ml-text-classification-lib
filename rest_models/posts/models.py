from django.db import models


# Create your models here.
class Post(models.Model):
    text = models.CharField(max_length=512)
    offensive_rating = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)
