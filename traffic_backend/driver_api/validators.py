import numpy as np
import json
from django.core.exceptions import ValidationError
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.utils.translation import gettext_lazy as _

SIMILARITY_THRESHOLD = 0.55 

def validate_embedding(embedding):
    from .models import Driver
    """Check if similar face exists in database"""
    drivers = Driver.objects.all().only("id", "embedding", 'license_number')
    best_match = None
    highest_similarity = -1
    # for driver in drivers:
    #     db_embedding = np.array(json.loads(driver.embedding))
    #     similarity = np.dot(embedding, db_embedding)
        
        
    #     if similarity > SIMILARITY_THRESHOLD:
    #         raise ValidationError(
    #             _("A driver with a similar face already exists: %(full_name)s (License No: %(license_number)s)"),
    #             params={
    #                 "full_name": f"{driver.first_name} {driver.middle_name} {driver.last_name}",
    #                 "license_number": driver.license_number
    #             },
    #         )
    for driver in drivers:
            try:
                db_embedding = np.array(json.loads(driver.embedding))
                similarity = np.dot(embedding, db_embedding)


                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = driver
            except Exception as e:
                continue

            if best_match and highest_similarity > SIMILARITY_THRESHOLD:
                raise ValidationError(
                    _("A driver with a similar face already exists: %(full_name)s (License No: %(license_number)s)"),
                    params={
                        "full_name": f"{driver.first_name} {driver.middle_name} {driver.last_name}",
                        "license_number": driver.license_number
                    },
                ) 


