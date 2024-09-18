from django.db import models
import json

class ScrapedData(models.Model):
    alias = models.CharField(max_length=100)
    url = models.URLField()
    title = models.CharField(max_length=255)
    scraped_at = models.DateTimeField()
    status = models.BooleanField(default=True)
    domain = models.CharField(max_length=255)
    all_anchor_href = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_anchors = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_images_data = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_images_source_data = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_h1_data = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_h2_data = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_h3_data = models.TextField(blank=True, null=True)  # Store JSON as a string
    all_p_data = models.TextField(blank=True, null=True)  # Store JSON as a string

    def save(self, *args, **kwargs):
        # Serialize JSON fields
        if isinstance(self.all_anchor_href, list):
            self.all_anchor_href = json.dumps(self.all_anchor_href)
        if isinstance(self.all_anchors, list):
            self.all_anchors = json.dumps(self.all_anchors)
        if isinstance(self.all_images_data, list):
            self.all_images_data = json.dumps(self.all_images_data)
        if isinstance(self.all_images_source_data, list):
            self.all_images_source_data = json.dumps(self.all_images_source_data)
        if isinstance(self.all_h1_data, list):
            self.all_h1_data = json.dumps(self.all_h1_data)
        if isinstance(self.all_h2_data, list):
            self.all_h2_data = json.dumps(self.all_h2_data)
        if isinstance(self.all_h3_data, list):
            self.all_h3_data = json.dumps(self.all_h3_data)
        if isinstance(self.all_p_data, list):
            self.all_p_data = json.dumps(self.all_p_data)
        super().save(*args, **kwargs)

    def load_json_field(self, field_name):
        value = getattr(self, field_name)
        if value:
            return json.loads(value)
        return []

    def __str__(self):
        return self.alias
