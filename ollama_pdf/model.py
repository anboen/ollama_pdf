from pydantic import BaseModel, Field
from typing import Optional


class AmountModel(BaseModel):
    total: float = Field(description="The total amount of the invoice")
    currency: str = Field(description="The currency of the amounts")


class TaxModel(BaseModel):
    amount: float = Field(description="The amount of the tax")
    rate: float = Field(description="The rate of the tax")


class AddressModel(BaseModel):
    street: str = Field(
        description="The street of the address of the invoice issuer"
    )
    city: str = Field(
        description="The city of the address of the invoice issuer"
    )
    zip_code: str = Field(
        description="The zip code (PLZ) of the address of the invoice issuer"
    )


class InvoiceResponse(BaseModel):
    """Extracted invoice data from a PDF document."""

    document_title: str = Field(
        description="The generated title of the invoice"
    )
    date: Optional[str] = Field(
        description="The date of the invoice in the format DD.MM.YYYY,"
        " i.e. 02.03.2005"
    )
    reference: Optional[str] = Field(
        None, description="The reference number of the invoice"
    )
    amounts: Optional[list[AmountModel]] = Field(
        description="The amounts of the invoice"
    )
    taxes: Optional[list[TaxModel]] = Field(
        description="The taxes of the invoice if applicable. "
        "Only report taxes which are more than 0.0"
    )
    address: Optional[AddressModel] = Field(
        description="The address of the invoice issuer"
    )
    IBAN: Optional[str] = Field(description="The IBAN of the invoice issuer")
    errors: Optional[list[str]] = Field(
        None,
        description="List of errors that occurred during the extraction. "
        "If there are no errors, this field is not present.",
    )
