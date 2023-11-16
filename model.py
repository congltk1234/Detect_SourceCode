from pydantic import BaseModel
from typing import Union

class Item(BaseModel):
    text: str