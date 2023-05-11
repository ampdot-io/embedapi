import random
import string

import numpy as np

import embedapi


def test_model(model):
    s = generate_random_string()
    r1 = embedapi.encode_one(model, s)
    r2 = embedapi.encode_one(model, s)
    assert r1.all() == r2.all()
    assert isinstance(r1, np.ndarray)
    assert isinstance(r2, np.ndarray)
    s2 = generate_random_string()
    r3 = embedapi.encode_batch(model, [s, s2])
    assert r3[0].all() == r1.all()
    assert isinstance(r3[1], np.ndarray)
    r4 = embedapi.encode_batch(model, [s, s2])
    assert r3.all() == r4.all()
    assert isinstance(r4, np.ndarray)


def generate_random_string():
    return ''.join(random.choices(string.printable, k=30))


if __name__ == '__main__':
    test_model('intfloat/e5-large')
    test_model('sentence-transformers/all-mpnet-base-v2')
