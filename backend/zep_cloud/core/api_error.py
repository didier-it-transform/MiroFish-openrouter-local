class ApiError(Exception):
    """Compatibility exception mirroring the Zep Cloud SDK ApiError."""

    def __init__(self, status_code=None, body=None, headers=None, message=None):
        if message is None:
            if isinstance(body, dict):
                message = body.get("message") or body.get("error") or "API error"
            else:
                message = str(body) if body is not None else "API error"
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.headers = headers if isinstance(headers, dict) else None
