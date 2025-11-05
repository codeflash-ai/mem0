from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class S3VectorsConfig(BaseModel):
    vector_bucket_name: str = Field(description="Name of the S3 Vector bucket")
    collection_name: str = Field("mem0", description="Name of the vector index")
    embedding_model_dims: int = Field(1536, description="Dimension of the embedding vector")
    distance_metric: str = Field(
        "cosine",
        description="Distance metric for similarity search. Options: 'cosine', 'euclidean'",
    )
    region_name: Optional[str] = Field(None, description="AWS region for the S3 Vectors client")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Precompute allowed_fields only once per class for efficiency
        if not hasattr(cls, "_allowed_fields_cache"):
            # Use tuple instead of set for fixed allowed_fields; should be small and membership is still fast
            cls._allowed_fields_cache = tuple(cls.model_fields.keys())
        allowed_fields = cls._allowed_fields_cache
        # Filter extra fields faster using set difference
        extra_fields = set(values) - set(allowed_fields)
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(sorted(extra_fields))}. Please input only the following fields: {', '.join(sorted(allowed_fields))}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
