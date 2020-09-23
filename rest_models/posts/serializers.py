from rest_framework import serializers
from .models import Post

import sys
sys.path.append('/Users/dawidbrzozowski/Projects/offensive-language-semeval')
from predictors import Predictor
from utils.files_io import load_json

predictor = Predictor(load_json('configs/data/predictor_config.json'))


class PostSerializer(serializers.ModelSerializer):

    class Meta:
        model = Post
        fields = (
            'text',
            'offensive_rating'
        )
        read_only_fields = (
            'offensive_rating',
        )

    def create(self, validated_data):
        validated_data['offensive_rating'] = predictor.predict(validated_data['text'])[0][0]
        return Post.objects.create(**validated_data)
