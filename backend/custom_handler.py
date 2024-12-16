import json
from generate import generate_email

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        result = generate_email(body)

        return {
            "statusCode": 200,
            "body": json.dumps(result),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
