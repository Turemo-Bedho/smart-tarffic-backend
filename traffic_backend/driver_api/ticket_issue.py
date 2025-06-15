# traffic/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Ticket # Ensure Ticket model is imported
# Import your SMS function
from .utils import send_ticket_sms # Assuming send_ticket_sms is in traffic/utils.py

@receiver(post_save, sender=Ticket)
def create_ticket_sms_notification(sender, instance, created, **kwargs):
    """
    Sends an SMS when a new Ticket is created.
    """
    if created: # Only send SMS when a new Ticket is created (not updated)
        driver = instance.violation.driver
        
        send_ticket_sms("", instance.reference_code, driver.first_name + " " + driver.last_name, instance.violation.location)
        print(f"SMS sent for ticket {instance.reference_code} to {driver.phone_number}")
        