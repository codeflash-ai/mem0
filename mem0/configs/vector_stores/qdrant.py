from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QdrantConfig(BaseModel):
    from qdrant_client import QdrantClient

    QdrantClient: ClassVar[type] = QdrantClient

    collection_name: str = Field("mem0", description="Name of the collection")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensions of the embedding model")
    client: Optional[QdrantClient] = Field(None, description="Existing Qdrant client instance")
    host: Optional[str] = Field(None, description="Host address for Qdrant server")
    port: Optional[int] = Field(None, description="Port for Qdrant server")
    path: Optional[str] = Field("/tmp/qdrant", description="Path for local Qdrant database")
    url: Optional[str] = Field(None, description="Full URL for Qdrant server")
    api_key: Optional[str] = Field(None, description="API key for Qdrant server")
    on_disk: Optional[bool] = Field(False, description="Enables persistent storage")

    @model_validator(mode="before")
    @classmethod
    def check_host_port_or_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Use local variables for faster attribute access and tuple unpacking for efficiency
        host = values.get("host")
        port = values.get("port")
        path = values.get("path")
        url = values.get("url")
        api_key = values.get("api_key")
        # Combine conditions into a single branch to reduce branching cost
        # No change in exception raising logic or message
        if path is None and (host is None or port is None) and (url is None or api_key is None):
            raise ValueError("Either 'host' and 'port' or 'url' and 'api_key' or 'path' must be provided.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
