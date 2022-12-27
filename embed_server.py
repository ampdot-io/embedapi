import io
from typing import List

import numpy as np

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from functools import lru_cache
from numpy.lib import format


# todo: autoboot and scale to zero

class EncodeRequest(BaseModel):
    input: List[str]


app = FastAPI()

transformer = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(transformer)


def yield_from_file(file):
    # yield from file will not wor
    yield file.getvalue()

# todo: better cache
encode = lru_cache(lambda x: model.encode(x))


@app.post(f"/{transformer}")
async def root(encode_request: EncodeRequest):
    # fixme: support sending as binary data
    embeddings = encode(tuple(encode_request.input))
    virt_file = io.BytesIO()
    np.save(virt_file, embeddings, allow_pickle=False)
    return StreamingResponse(yield_from_file(virt_file))
