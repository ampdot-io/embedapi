import embedapi

if __name__ == '__main__':
    embedapi.encode_one('intfloat/e5-large', 'hello world')
    embedapi.encode_one('sentence-transformers/all-mpnet-base-v2', 'hello world')

