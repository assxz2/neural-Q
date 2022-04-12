import json
import pathlib
from pathlib import Path

import connexion
import neotime
import os
from py2neo import Node, Relationship

from swagger_server import celery_tasks, db_connection
from swagger_server.models.inline_response201 import InlineResponse201  # noqa: E501
from swagger_server.models.project import Project  # noqa: E501


def db_project_find(project_model):
    """Find the project in the database and return node

    :param project_model: project model
    :type project_model: object
    """
    graph, matcher = db_connection()
    project = matcher.match('Project', creator=project_model.author_name, title=project_model.name).first()
    return project


def db_project_create(project_model):
    """Create Project link the user, samples

    :return: If project created successfully
    :rtype: bool
    """
    graph, matcher = db_connection()
    project_path = pathlib.Path(project_model.author_name).joinpath(project_model.name)
    project = Node('Project',
                   title=project_model.name,
                   creator=project_model.author_name,
                   path=str(project_path),
                   description=project_model.remark_text,
                   timestamp=str(neotime.DateTime.now()))

    # create project with no sample is forbidden
    if len(project_model.sample_urls) == 0:
        return False

    # test samples exists status
    valid_sample_nodes = []
    for sample in project_model.sample_urls:
        sample_node = matcher.match('Sample', title=sample).first()
        if sample_node is not None:
            valid_sample_nodes.append(sample_node)
        else:
            return False

    user = matcher.match('User', username=project_model.author_name).first()
    u2project = Relationship(user, 'CREATE', project)
    graph.create(u2project)
    for sample_node in valid_sample_nodes:
        rel_sample2project = Relationship(sample_node, 'IMPORT', project)
        graph.create(rel_sample2project)
    return True


def create_project(body):  # noqa: E501
    """Create a project by given parameters

     # noqa: E501

    :param body: Project object that needs to be added to the cloud
    :type body: dict | bytes

    :rtype: InlineResponse201
    """
    if connexion.request.is_json:
        body = Project.from_dict(connexion.request.get_json())  # noqa: E501
        if db_project_find(body) is not None:
            return 'Project name exists!', 400
        if db_project_create(body):
            celery_tasks.create_project(body.name, body.author_name)
            return 'OK', 201
        else:
            return 'At least 1 video is required', 406
    else:
        return 'invaled project', 405



def file_tree_show(location):  # noqa: E501
    """get file tree from current dir

     # noqa: E501

    :param location: Name of project to add
    :type location: str

    :rtype: FileTree
    """
    filepath = location
    result = os.listdir(filepath)
    if len(result) > 0:
        return result
    else:
        return []


