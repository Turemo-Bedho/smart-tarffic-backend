from django.conf import settings
from twilio.rest import Client

def send_ticket_sms(phone_number, reference_code, name, location):
    """
    Sends an SMS notification with the ticket reference code.
    
    Args:
        phone_number (str): The phone number to send the SMS to.
        reference_code (str): The ticket reference code to include in the SMS.
    """
    # This is a placeholder for your actual SMS sending logic.
    # You would typically use an external service like Twilio, Nexmo, etc.
    message = f"""
    Dear {name},
    You have received an Official Traffic Violation Notice. Your ticket reference code is {reference_code} for a violation at {location}.
    """
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to="+251945359970"
        )
        print(f"SMS sent successfully: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False  # Return False if the SMS failed to send


    return True  # Return True if the SMS was sent successfully