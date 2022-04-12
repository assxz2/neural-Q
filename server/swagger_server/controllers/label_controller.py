from os import link
import connexion
import six
import neotime

from datetime import datetime
from py2neo import Node, Relationship
from swagger_server.models.rectangle import Rectangle  # noqa: E501
from swagger_server import util, db_connection


def db_label_rectangle_create(rect_model, project_name, username, sample_name):
    """Create rectangle in database and link to corresponding project and sample.

    :param rect_model: rectangle model body
    :type rect_model: Rectangle Model
    :param project_name: corresponding project name [title in db]
    :type project_name: str
    :param username: username
    :type username: str
    :param sample_name: corresponding sample name
    :type sample_name: str
    """
    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()
    sample = matcher.match('Sample', title=sample_name).first()

    # If project not exists, everthing has no meaning
    if project is None:
        print("Project not found!")
        return False

    # If sample not exists, annotation has no meaning
    if sample is None:
        print("Sample not found!")
        return False

    # Check if sample has import into current project
    rel = graph.match_one((sample, project), r_type='IMPORT')
    if rel is None:
        print("Sample not in current Project!")
        return False

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    rectangle_filename = '-'.join(['rectangle', suffix])
    rect = Node('Annotation',
                title=rect_model.name,
                label_format='Rectangle',
                creator=username,
                top_left_x=rect_model.top_left_x,
                top_left_y=rect_model.top_left_y,
                bottom_right_x=rect_model.bottom_right_x,
                bottom_right_y=rect_model.bottom_right_y,
                path=f"{project['path']}/{rectangle_filename}",
                timestamp=str(neotime.DateTime.now()))
    link_sample = Relationship(rect, 'ANNOTATE', sample)
    graph.create(link_sample)
    link_project = Relationship(project, 'BRING', rect)
    graph.create(link_project)

    return True


def db_create_label_skeleton(skeleton_file, project_name, username, sample_name, region_name):
    """Create skeleton annotation and link to corresponding project and sample region

    :param skeleton_file: Uploaded file object
    :type skeleton_file: FileStorage
    :param project_name: project name
    :type project_name: str
    :param username: username
    :type username: str
    :param sample_name: sample name (title in database)
    :type sample_name: str
    :param region_name: region name corresponding to rectangle annotation autotracked and clipped video clip
    :type region_name: str
    """
    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()
    # If project not exists, everything has no meaning
    if project is None:
        print("Project not found!")
        return False

    region_result = matcher.match('Result', path=f"{project['path']}/{region_name}").first()
    # If sample not exists, annotation has no meaning
    if region_result is None:
        print("Region result not found!")
        return False

    skeleton_path = f"{region_result['path']}/skeleton.csv"
    skeleton_file.save(f"/workspace/{skeleton_path}")

    # If skeleton annotation not exist, create one
    # else update the timestamp of current skeleton node
    skeleton = matcher.match('Annotation', path=skeleton_path).first()
    if skeleton is None:
        skeleton = Node('Annotation',
                        title='Landmarks.csv',
                        label_format='Skeleton',
                        creator=username,
                        path=skeleton_path,
                        timestamp=str(neotime.DateTime.now()))
        graph.create(skeleton)
        link_project = Relationship(project, 'BRING', skeleton)
        graph.create(link_project)
        link_region = Relationship(skeleton, 'ANNOTATE', region_result)
        graph.create(link_region)
    else:
        skeleton.update({
            'timestamp': str(neotime.DateTime.now())
        })
        graph.push(skeleton)
    return True


def submit_rectangle(project_name, username, sample_name, body=None):  # noqa: E501
    """Add the annotation to frame

    Add a annotation to frame # noqa: E501

    :param project_name: Name of project to add
    :type project_name: str
    :param username: Name of user
    :type username: str
    :param sample_name: Name of frame to add
    :type sample_name: str
    :param body:
    :type body: dict | bytes

    :rtype: Dict[str, int]
    """
    if connexion.request.is_json:
        body = Rectangle.from_dict(connexion.request.get_json())  # noqa: E501
        if db_label_rectangle_create(body, project_name, username, sample_name):
            return 'OK', 201
    return 'do some magic!'


def submit_skeleton(project_name, username, sample_name, region_name, file_name=None):  # noqa: E501
    """Add the skeletons collection file to sample or sample clip

    Add a annotation to frame # noqa: E501

    :param project_name: Name of project to add
    :type project_name: str
    :param username: Name of user
    :type username: str
    :param sample_name: Name of frame to add
    :type sample_name: str
    :param region_name: Target ROI name for current skeleton
    :type region_name: str
    :param file_name:
    :type file_name: strstr

    :rtype: Dict[str, int]
    """
    uploaded_file = connexion.request.files['fileName']
    if db_create_label_skeleton(uploaded_file, project_name, username, sample_name, region_name):
        return 'OK', 201
    return 'do some magic!'


def update_skeleton(project_name, username, sample_name, region_name, file_name=None):  # noqa: E501
    """Add the annotation to frame

    Add a annotation to frame # noqa: E501

    :param project_name: Name of project to add
    :type project_name: str
    :param username: Name of user
    :type username: str
    :param sample_name: Name of frame to add
    :type sample_name: str
    :param region_name: Target ROI name for current skeleton
    :type region_name: str
    :param file_name: 
    :type file_name: strstr

    :rtype: Dict[str, int]
    """
    return 'do some magic!'
