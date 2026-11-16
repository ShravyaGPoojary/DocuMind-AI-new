import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z0-9\.\-]+$")


class UserRegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    email: str = Field(..., description="Valid unique email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (at least 6 characters)")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="Account email address")
    password: str = Field(..., min_length=1, description="Account password")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean



class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., min_length=10, description="Google OAuth ID Token from Google Identity Services")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    avatar_url: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    message: str
