from pydantic import BaseModel, Field
from typing import Optional, Union


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
    date: str = Field(
        description="The date of the invoice in the format DD.MM.YYYY,"
        " i.e. 02.03.2005"
    )
    reference: Optional[str] = Field(
        None, description="The reference number of the invoice"
    )
    amounts: list[AmountModel] = Field(
        description="The amounts of the invoice"
    )
    taxes: Optional[list[TaxModel]] = Field(
        description="The taxes of the invoice if applicable. "
        "Only remport taxes which are more thant 0.0"
    )
    address: AddressModel = Field(
        description="The address of the invoice issuer"
    )
    IBAN: str = Field(description="The IBAN of the invoice issuer")


class ErrorResponse(BaseModel):
    message: str = Field(
        description="Error message if the PDF couldn't be processed"
    )


class FinalResponse(BaseModel):
    """Final response can either be an InvoiceResponse or an ErrorResponse.
    Choose ErrorResponse if the PDF isn't a valid invoice."""

    response: Union[InvoiceResponse, ErrorResponse]


"""RESPOSNE_SCHEMA = {
    "title": "Invoice",
    "description": "An invoice recieved in PDF",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The generated title of the invoice",
        },
        "date": {
            "type": "string",
            "description": "The date of the invoice in the format DD.MM.YYYY",
        },
        "reference": {
            "type": "string",
            "description": "The reference number of the invoice",
        },
        "amounts": {
            "type": "array",
            "description": "The amounts of the invoice",
            "minItems": 1,
            "items": {
                "type": "object",
                "description": "The amount of the invoice",
                "properties": {
                    "total": {
                        "type": "number",
                        "description": "The total amount of the invoice",
                    },
                    "currency": {
                        "type": "string",
                        "description": "The currency of the amounts",
                    },
                },
                "required": ["total", "currency"],
            },
        },
        "taxes": {
            "type": "array",
            "description": "The taxes of the invoice if applicable",
            "minItems": 0,
            "maxItems": 3,
            "items": {
                "type": "object",
                "description": "The tax of the invoice",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount of the tax",
                    },
                    "rate": {
                        "type": "number",
                        "description": "The rate of the tax",
                    },
                },
                "required": ["amount", "rate"],
            },
        },
        "address": {
            "type": "object",
            "description": "The address of the invoice issuer",
            "properties": {
                "street": {
                    "type": "string",
                    "description": "The street of the address",
                },
                "city": {
                    "type": "string",
                    "description": "The city of the address",
                },
                "zip_code": {
                    "type": "string",
                    "description": "The zip code of the address",
                },
            },
            "required": ["street", "city", "zip_code"],
        },
        "IBAN": {
            "type": "string",
            "description": "The IBAN of the invoice issuer",
        },
        "ERROR": {
            "type": "string",
            "description": "Error message if the PDF couldn't be processed",
        },
    },
}"""
