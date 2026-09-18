import json
def main(arg1: str) -> dict:
    try:
        data = json.loads(arg1)
        return data
    except Exception:
        return {
            "confirm": "false",
            "other": "false"
        }


if __name__ == '__main__':
    arg1 = '{"email": "admin@example.com", "password": "admin_password"}'
    print(main(arg1))


