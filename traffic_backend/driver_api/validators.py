import numpy as np
import json
from django.core.exceptions import ValidationError
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.utils.translation import gettext_lazy as _

SIMILARITY_THRESHOLD = 0.65  

def validate_embedding(embedding):
    from .models import Driver
    """Check if similar face exists in database"""
    
    for driver in Driver.objects.all():
        db_embedding = np.array(json.loads(driver.embedding))
        similarity = np.dot(embedding, db_embedding)
        
        
        if similarity > SIMILARITY_THRESHOLD:
            raise ValidationError(
                _("A driver with a similar face already exists: %(full_name)s (License No: %(license_number)s)"),
                params={
                    "full_name": f"{driver.first_name} {driver.middle_name} {driver.last_name}",
                    "license_number": driver.license_number
                },
            )