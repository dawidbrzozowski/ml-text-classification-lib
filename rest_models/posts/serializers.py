from rest_framework import serializers
from .models import Post
from .apps import PostsConfig


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
        validated_data['offensive_rating'] = PostsConfig.predictor.predict(validated_data['text'])[0][0]
        return Post.objects.create(**validated_data)
