import io
import json
import time

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
    # possible alternative: vary by token count
    # https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
    for chunk in more_itertools.chunked(data, 493):  # 493 = e ** 6
        for retry in range(6):
            try:
                response = openai.Embedding.create(
                    model=transformer,
                    input=chunk
                )
            except (json.decoder.JSONDecodeError, openai.error.APIConnectionError) as e:
                time.sleep(2 ** (retry - 1))
            except openai.error.RateLimitError as e:
                time.sleep(2 ** (retry - 1))
            else:
                result.extend(response['data'])
                break
        else:
            raise e
    return np.array([obj["embedding"] for obj in result], dtype=np.float32)


def is_openai_model(s):
    return 'ada' in s or 'babbage' in s or 'curie' in s or 'cushman' in s or 'davinci' in s

# todo: add tests that check .shape
# ada is 1536
# all-mpnet-base-v2 is 768

