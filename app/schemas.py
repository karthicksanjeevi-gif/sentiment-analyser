from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class AnalyzeTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


class RAGRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


class SourceItem(BaseModel):
    source: str
    score: float
    excerpt: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class WordFrequency(BaseModel):
    word: str
    count: int


class AnalyzedReview(BaseModel):
    text: str
    label: str
    confidence: float


class DocumentAnalysisResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    positive_percentage: float
    negative_percentage: float
    average_confidence: float
    positive_top_words: list[WordFrequency]
    negative_top_words: list[WordFrequency]
    positive_confidence_distribution: dict[str, int]
    negative_confidence_distribution: dict[str, int]
    reviews: list[AnalyzedReview]

