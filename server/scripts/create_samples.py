import os

import cv2
import neotime
from py2neo import Graph, NodeMatcher, Node, Relationship

neo4j_url = os.environ.get('NEO4J_URL', 'http://neo4j:7474/db/data/')
neo4j_username = os.environ.get('NEO4J_USERNAME', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'neo4jtest2020')
graph = Graph(neo4j_url, user=neo4j_username, password=neo4j_password)
matcher = NodeMatcher(graph)


def scan_directory_and_create_samples(data_dir, data_prefix, creator_node, taxonomy_node):
    """Scan and check if data exists

    :param data_dir: Root dir for scanning data
    :type data_dir: str
    :param data_prefix: title prefix for construct its title in database
    :type data_prefix: str
    :param creator_node: neo4j node of sample create user
    :type creator_node: Node
    :param taxonomy_node: neo4j node of sample taxonomy_node
    :type taxonomy_node: Node
    :return:
    """
    for item in os.listdir(data_dir):
        if item[-4:] == '.avi':
            item_title = os.path.join(data_prefix, item)
            item_path = os.path.join(data_dir, item)
            item_node = matcher.match('Sample', title=item_title).first()
            if item_node is not None:
                print("FOUND", item_node['title'])
            if item_node is None:
                cap = cv2.VideoCapture(item_path)
                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, frame = cap.read()
                cv2.imwrite(item_path[:-4] + '.png', frame)
                cap.release()
                item_node = Node('Sample',
                                 title=item_title,
                                 creator=creator_node['username'],
                                 description="An OK sample",
                                 taxonomy=taxonomy_node['title'],
                                 cover_path=item_path[:-4] + '.png',
                                 tag='GAL4',
                                 num_frame=num_frame,
                                 timestamp=str(neotime.DateTime.now()))
                link_creator = Relationship(creator_node, 'RECORD', item_node)
                graph.create(link_creator)
                link_taxonomy = Relationship(item_node, 'INSTANCE_OF', taxonomy_node)
                graph.create(link_taxonomy)
                print("CREATE Sample: {}".format(item_title))
    return True


if __name__ == '__main__':
    user_node = matcher.match('User', username='sunyixuan').first()
    taxonomy_node = matcher.match('Taxonomy', title='Drosophila/VNC').first()
    scan_directory_and_create_samples('/data/drosophila/light-sheet/VNC',
                                      'drosophila/light-sheet/VNC', user_node, taxonomy_node)
