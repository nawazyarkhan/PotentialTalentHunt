from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    in_stock: bool = True


@app.get("/")
def home():
    return {"message": "FastAPI inside conda works!"}


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/items/")
def create_item(item: Item):
    return {
        "message": "Item created",
        "item": item
    }


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {
        "message": "Item updated",
        "item_id": item_id,
        "item": item
    }
