# Generated by Django 5.1.1 on 2024-09-18 04:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='scrapeddata',
            name='all_anchor_href',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_anchors',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_h1_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_h2_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_h3_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_images_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_images_source_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='scrapeddata',
            name='all_p_data',
            field=models.TextField(blank=True, null=True),
        ),
    ]