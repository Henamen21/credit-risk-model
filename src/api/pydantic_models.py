from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):

    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str  # or datetime if you plan to parse it
    PricingStrategy: int
    FraudResult: Optional[int] = None  # Optional  # Optional if you're predicting this; else include for training


class PredictionResponse(BaseModel):
    risk_probability: float
