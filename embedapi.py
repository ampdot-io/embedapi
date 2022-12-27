import io

import requests
import numpy as np
import more_itertools

base_url = 'http://localhost:7000/{}'

# todo: nptyping
def encode_one(transformer: str, datum: str) -> np.array:
    assert isinstance(datum, str)
    return encode_batch(transformer, [datum])[0]

def encode_batch(transformer: str, data: list[str]) -> np.array:
    if is_openai_model(transformer):
        return _openai_encode_batch(transformer, data)
    else:
        return _local_encode_batch(transformer, data)

def _local_encode_batch(transformer: str, data: list[str]) -> np.array:
    response = requests.post(
        base_url.format(transformer),
        json={"input": data}
    )
    response.raise_for_status()
    virt_file = io.BytesIO(response.content)
    data = np.load(virt_file)
    return data


def _openai_encode_batch(transformer: str, data: list[str]) -> np.array:
    import openai
    assert transformer.startswith('text-embedding')
    result = []
    for chunk in more_itertools.chunked(data, 1000):
        response = openai.Embedding.create(
            model=transformer,
            input=chunk
        )
        result.extend(response['data'])
    return np.array([obj["embedding"] for obj in result], dtype=np.float32)


def is_openai_model(s):
    return 'ada' in s or 'babbage' in s or 'curie' in s or 'cushman' in s or 'davinci' in s

# todo: add tests that check .shape
# ada is 1536
# all-mpnet-base-v2 is 768

