import os

import celery
from py2neo import NodeMatcher, Graph

NEO4J_URL = os.environ.get('NEO4J_URL', 'http://neo4j:7474')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neo4jtest2020')

# graph.schema.create_uniqueness_constraint('Annotation', 'path')
# graph.schema.create_uniqueness_constraint('Sample', 'title')
# graph.schema.create_uniqueness_constraint('User', 'username')
# graph.schema.create_uniqueness_constraint('Taxonomy', 'title')
# graph.schema.create_uniqueness_constraint('Model', 'path')

celery_app = celery.Celery('rouran Worker')
celery_app.config_from_object('swagger_server.celery_config')


def db_connection():
    """

    :return (graph, matcher):
    """
    graph = Graph('http://neo4j:7474/', user='neo4j', password='neo4jtest2020')
    matcher = NodeMatcher(graph)
    return graph, matcher