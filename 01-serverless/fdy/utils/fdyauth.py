import os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# DOTENV = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
# print(f"DOTENV file path: {DOTENV}")

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings are validated through Pydantic's type system.
    """
    project_name: str = Field(..., env="PROJECT_NAME")
    resource_group_name: str = Field(..., env="RESOURCE_GROUP_NAME")
    subscription_id: str = Field(..., env="SUBSCRIPTION_ID")

    class Config:
        """Pydantic model configuration"""
        # env_file = f"{DOTENV}"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False # variable names are lowcase in the Settings

class AuthHelper:
    @classmethod
    def load_settings(cls):
        try:
            return Settings()
        except ValidationError as e:
            missing = [err['loc'][0] for err in e.errors()]
            raise EnvironmentError(f"Missing required env keys: {missing}")

    @staticmethod
    def test_credential():
        """
        Test Azure authentication and return a credential object.
        Falls back to InteractiveBrowserCredential if DefaultAzureCredential fails.
        """
        credential = None
        try:
            credential = DefaultAzureCredential()
            # credential.get_token("https://management.azure.com/.default")
            credential.get_token("https://ai.azure.com/.default")
            # print("DefaultAzureCredential authentication OK")
        except Exception:
            credential = InteractiveBrowserCredential()
            # print("Falling back to InteractiveBrowserCredential")
        return credential
    
# import pydantic 
# print(f"Pydantic version: {pydantic.VERSION}")