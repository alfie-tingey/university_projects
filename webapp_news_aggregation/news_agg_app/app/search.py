from flask import current_app
from flask_babel import _, get_locale

def add_to_index(index, model):
    ''' add to search index '''
    if not current_app.elasticsearch:
        return
    payload = {}
    for field in model.__searchable__:
        payload[field] = getattr(model, field)
    current_app.elasticsearch.index(index=index, id=model.id, body=payload)

def remove_from_index(index, model):
    ''' remove from search index '''
    if not current_app.elasticsearch:
        return
    current_app.elasticsearch.delete(index=index, id=model.id)

def query_index(index, query, page, per_page):
    ''' Find the best matches for the expression in the full-text search'''

    if not current_app.elasticsearch:
        return [], 0
    search = current_app.elasticsearch.search(
        index=index,
        body={'query': {'multi_match': {'query': query, 'fields': ['*']}},
              'from': (page - 1) * per_page, 'size': per_page})

    h = search['hits']['hits']

    ids = [(hit['_id']) for hit in search['hits']['hits']]
    return ids, search['hits']['total']
