from datetime import datetime

import connexion
import neotime
from py2neo import Node, Relationship

from swagger_server import celery_tasks, db_connection
from swagger_server.models.rectangle import Rectangle  # noqa: E501


def db_create_tracking_inference(rect_model, project_name, username, sample_name, model_path):
    """Inference the region sequence with corresponding rectangle on sample cover.

    :param rect_model: rectangle model body
    :type rect_model: Rectangle Model
    :param project_name: corresponding project name [title in db]
    :type project_name: str
    :param username: username
    :type username: str
    :param sample_name: corresponding sample name
    :type sample_name: str
    :param model_path: corresponding model path
    :type model_paht: str
    """

    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()
    sample = matcher.match('Sample', title=sample_name).first()
    model = matcher.match('Model', path=model_path).first()

    # If project not exists, everthing has no meaning
    if project is None:
        print("Project not found!")
        return False

    # If sample not exists, annotation has no meaning
    if sample is None:
        print(sample_name)
        print("Sample not found!")
        return False

    # Check if sample has import into current project
    rel = graph.match_one((sample, project), r_type='IMPORT')
    if rel is None:
        print("Sample not in current Project!")
        return False

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    rectangle_filename = '-'.join([rect_model.name, 'rectangle', suffix])

    query = Node('Query',
                 title=rect_model.name,
                 label_format='Rectangle',
                 creator=username,
                 top_left_x=rect_model.top_left_x,
                 top_left_y=rect_model.top_left_y,
                 bottom_right_x=rect_model.bottom_right_x,
                 bottom_right_y=rect_model.bottom_right_y,
                 path=f"{project['path']}/{rectangle_filename}",
                 timestamp=str(neotime.DateTime.now()))
    result = Node('Result',
                  title=rectangle_filename,
                  label_format='Rectangle',
                  model=model['title'],
                  status='Created',
                  sample=sample_name,
                  path=f"{project['path']}/{rectangle_filename}",
                  csv_path=f"{project['path']}/{rectangle_filename}/tracking.csv",
                  clip_path=f"{project['path']}/{rectangle_filename}/clip.tif",
                  timestamp=str(neotime.DateTime.now()))
    link_project = Relationship(project, 'EXTRACT', result)
    graph.create(link_project)
    link_query = Relationship(query, 'ASK_FOR', result)
    graph.create(link_query)
    link_sample = Relationship(sample, 'HAS_INFO', result)
    graph.create(link_sample)
    link_model = Relationship(model, 'INFERENCE', result)
    graph.create(link_model)

    celery_tasks.inference_tracking.delay(query, result)

    reply = {
        'name': rect_model.name,
        'uri': result['path']
    }
    return reply


def db_create_landmark_inference(project_name, username, sample_name, region_name):
    """Inference the skeleton sequence of queried sample region.

    :param project_name: Project name
    :type project_name: str
    :param username: username
    :type username: str
    :param sample_name: Sample name
    :type sample_name: str
    :param region_name: Region of interest name
    :type region_name: str
    """
    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()

    # If project not exists, everything has no meaning
    if project is None:
        print("Project not found!")
        return False

    if region_name is not None:
        model = matcher.match('Model', path=f"{project['path']}/{region_name}/models/finetuned_skeleton_model").first()
        data = matcher.match('Result', path=f"{project['path']}/{region_name}").first()
        # If data not exists, annotation has no meaning
        if data is None:
            print("Inference data not found!")
            return False
        data_path = data['clip_path']
        skeleton_filename = f"{region_name}/results"
    else:
        model = matcher.match('Model', path=f"{project['path']}/models/finetuned_skeleton_model").first()
        data = matcher.match('Sample', title=sample_name).first()
        if data is None:
            print("Inference data not found!")
            return False
        data_path = data['title']
        skeleton_filename = f"results"

    if model is None:
        print("Inference engine not created!")
        return False

    if model['status'] != 'Ready':
        print("Inference engine not Ready!")
        return False

    result = matcher.match('Result', path=f"{project['path']}/{skeleton_filename}").first()
    if result is None:
        result = Node('Result',
                      title='Skeleton',
                      label_format='Points',
                      model=model['title'],
                      status='Created',
                      data_path=data_path,
                      path=f"{project['path']}/{skeleton_filename}",
                      csv_path=f"{project['path']}/{skeleton_filename}/predicted_skeleton.csv",
                      rendered_path=f"{project['path']}/{skeleton_filename}/skeleton_rendered.avi",
                      timestamp=str(neotime.DateTime.now()))
        graph.create(result)
        link_project = Relationship(project, 'EXTRACT', result)
        graph.create(link_project)
        link_data = Relationship(data, 'HAS_INFO', result)
        graph.create(link_data)
        link_model = Relationship(model, 'INFERENCE', result)
        graph.create(link_model)
    if result['status'] != 'Ready':
        celery_tasks.inference_landmark.delay(result, model)
    return True


def db_create_analyse(uploaded_file,project_name, username, sample_name, region_name):
    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()

    # If project not exists, everything has no meaning

    if project is None:
        print("Project not found!")
        return False

    if region_name is not None:
        data = matcher.match('Result', path=f"{project['path']}/{region_name}").first()
        # If data not exists, annotation has no meaning
        if data is None:
            print("Inference data not found!")
            return False
        sample = matcher.match('Sample', title=sample_name).first()
        if sample is None:
            print('16 bit sample not exist')
            return False
        sample_name = sample_name.split('.')[0] + "_16." + sample_name.split('.')[1]
        result_path = f"{region_name}/results"
    else:
        return False
    analysis_path = f"{data['path']}/analysis.csv"
    uploaded_file.save(f"/workspace/{analysis_path}")

    result = matcher.match('Result',title='Analysis', path=f"{project['path']}/{result_path}").first()
    if result is None:
        result = Node('Result',
                      title='Analysis',
                      status='Created',
                      data_path=sample_name,
                      analysis_path=analysis_path,
                      path=f"{project['path']}/{result_path}",
                      csv_path=f"{project['path']}/{result_path}/lines.csv",
                      pdf_path=f"{project['path']}/{result_path}/lines.pdf",
                      timestamp=str(neotime.DateTime.now()))
        graph.create(result)
        link_project = Relationship(project, 'EXTRACT', result)
        graph.create(link_project)
        link_data = Relationship(data, 'HAS_INFO', result)
        graph.create(link_data)
    if result['status'] != 'Ready':
        celery_tasks.inference_signals.delay(result)
    return True

def inference_landmark(project_name, username, sample_name, region_name=None):  # noqa: E501
    """Return a specific region inference

    Get a specific frame inference results # noqa: E501

    :param project_name: Name of specific project
    :type project_name: str
    :param username: Username
    :type username: str
    :param sample_name: Name of specific sample
    :type sample_name: str
    :param region_name: Name of specific region for specific sample which is generated by tracking algorithm
    :type region_name: str

    :rtype: None
    """
    if db_create_landmark_inference(project_name, username, sample_name, region_name):
        return 'OK', 202
    return 'Fail!', 400


def inference_tracking(project_name, username, sample_name, body=None):  # noqa: E501
    """Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri

     # noqa: E501

    :param project_name: Name of specific project
    :type project_name: str
    :param username: Name of specific user
    :type username: str
    :param sample_name: Name of specific sample
    :type sample_name: str
    :param body:
    :type body: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        body = Rectangle.from_dict(connexion.request.get_json())  # noqa: E501
        inference_create_status = db_create_tracking_inference(body, project_name, username, sample_name,
                                                               'deep_learning/tracking/CIResNet22_RPN.pth')
        if inference_create_status:
            return inference_create_status, 202
    return 'do some magic!'


def inference_landmark_status(uri):  # noqa: E501
    """Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri

     # noqa: E501

    :param uri: URI for specific node
    :type uri: str

    :rtype: None
    """
    graph, matcher = db_connection()
    node = matcher.match('Result', path=uri).first()
    if node is not None:
        status = node['status']
        if status == 'Ready':
            base_url = connexion.request.host.split(':')[0] + ':8090/' + node['path']
            result = {
                'url': base_url,
                'csv_url': base_url + '/predicted_skeleton.csv',
                'video_url': base_url + '/clip.tif'
            }
            return result, 200
        else:
            return status, 202
    return 'Failed', 400

def inference_tracking_status(uri):  # noqa: E501
    """Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri

     # noqa: E501

    :param uri: URI for specific node
    :type uri: str

    :rtype: None
    """
    graph, matcher = db_connection()
    node = matcher.match('Result', path=uri).first()
    if node is not None:
        status = node['status']
        if status == 'Ready':
            base_url = connexion.request.host.split(':')[0] + ':8090/' + node['path']
            result = {
                'url': base_url,
                'csv_url': base_url + '/tracking.csv',
                'video_url': base_url + '/clip.tif'
            }
            return result, 200
        else:
            return status, 202
    return 'Failed', 400

def inference_analyse(project_name, username, sample_name, region_name=None):
    uploaded_file = connexion.request.files['fileName']

    if db_create_analyse(uploaded_file,project_name, username, sample_name, region_name):
        return 'OK', 202

def inference_analyse_status(uri):
    graph, matcher = db_connection()
    node = matcher.match('Result', title='Analysis', path=uri).first()
    if node is not None:
        status = node['status']
        if status == 'Ready':
            base_url = connexion.request.host.split(':')[0] + ':8090/' + node['path']
            result = {
                'url': base_url,
                'csv_url': base_url + '/lines.csv',
                'pdf_url': base_url + '/lines.pdf'
            }
            return result, 200
        else:
            return status, 202
    return 'Failed', 400