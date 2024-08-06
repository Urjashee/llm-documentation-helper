import pinecone
from pinecone import ServerlessSpec

pc = pinecone.Pinecone()


def delete_pinecone_index(index_name='all'):
    # import pinecone_operations
    # pc = pinecone_operations.Pinecone()

    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes')
        for index in indexes:
            pc.delete_index(index)
        print('Done')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


def create_pinecone_index(index_name):
    # creating a new index
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )