from .models import DeviceToken
from firebase_admin import messaging

def send_fcm_notification(device_token, title, body, data=None, image_url=None):
    print(device_token)
    """
    Sends an FCM notification to a specific device.

    Args:
        device_token (str): The FCM registration token of the target device.
        title (str): The title of the notification.
        body (str): The body/content of the notification.
        data (dict, optional): A dictionary of custom key-value pairs to send
                               with the message. Defaults to None.
    """
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
            image=image_url
        ),
        data=data,
        token=device_token,
    )

    try:
        response = messaging.send(message)
        print(f"Successfully sent message: {response}")
        return True
    except Exception as e:
        print(f"Error sending message: {e}")
        return False
    

def notify_officers_about_detection(license_plate_info):
    """
    Sends a notification to all active officers when a license plate is detected.
    """
    # Retrieve all active device tokens.
    # If "officers" are defined by a specific user group or a flag on the User model,
    # you would filter DeviceToken.objects.filter(user__is_officer=True, is_active=True)
    # or similar, depending on your user model.
    device_tokens = DeviceToken.objects.filter()

    if not device_tokens.exists():
        print("No active device tokens found for any officer.")
        return False

    title = "Unrecognized driver detected!"
    # body = f"License plate '{license_plate_info['plate_number']}' detected at {license_plate_info['location']}."
    body = "With vechicle plate number: " + license_plate_info.get('plate_text', "Unknown")
    data = {}

    success_count = 0
    for device_token_obj in device_tokens:
        if send_fcm_notification(device_token_obj.token, title, body, data, image_url=license_plate_info.get('plate_image_url', None)):
            success_count += 1
        else:
            # Optionally, mark the token as inactive if sending fails repeatedly
            # device_token_obj.is_active = False
            # device_token_obj.save()
            pass # For now, just print error in send_fcm_notification

    return success_count > 0