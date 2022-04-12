import connexion
import neotime
from py2neo import Node, Relationship

from swagger_server import celery_tasks, db_connection
from swagger_server.models.learning_strategy import LearningStrategy  # noqa: E501


def db_create_landmark_learning(strategy_model, project_name, username, sample_name, region_name, meta_model_path):
    """Create a landmark estimation learning task.

    :param strategy_model: Model with some learning strategies
    :type strategy_model: Strategy model
    :param project_name: corresponding project name [title in db]
    :type project_name: str
    :param username: username
    :type username: str
    :param sample_name: corresponding sample name
    :type sample_name: str
    :param region_name: corresponding region sequence clip for training
    :param region_name: str
    :param meta_model_path: corresponding model path
    :type meta_model_path: str
    """
    graph, matcher = db_connection()
    project = matcher.match('Project', title=project_name, creator=username).first()
    sample = matcher.match('Sample', title=sample_name).first()
    meta_model = matcher.match('Model', path=meta_model_path).first()

    # If project not exists, everthing has no meaning
    if project is None:
        print("Project not found!")
        return False

    # If sample not exists, annotation has no meaning
    if sample is None:
        print("Sample not found!")
        return False

    if meta_model is None:
        print("Meta model not found!")
        return False

    if region_name is not None:
        learn_data = matcher.match('Result', path=f"{project['path']}/{region_name}").first()
        if learn_data is None:
            return False

        annotation = matcher.match('Annotation', path=f"{project['path']}/{region_name}/skeleton.csv").first()
        if annotation is None:
            return False

        model = Node('Model',
                     title=f"finetune_skeleton_model",
                     creator=username,
                     status='Created',
                     path=f"{project['path']}/{region_name}/models/finetuned_skeleton_model",
                     timestamp=str(neotime.DateTime.now()))
    else:
        learn_data = sample
        annotation = matcher.match('Annotation', path=f"{project['path']}/{region_name}/skeleton.csv").first()
        if annotation is None:
            return False

        model = Node('Model',
                     title=f"finetune_skeleton_model",
                     creator=username,
                     status='Created',
                     path=f"{project['path']}/models/finetuned_skeleton_model",
                     timestamp=str(neotime.DateTime.now()))

    try:
        link_data = Relationship(learn_data, 'FEED', model)
        graph.create(link_data)
        link_label = Relationship(annotation, 'FEED', model)
        graph.create(link_label)
        link_meta = Relationship(meta_model, 'FINE_TUNE', model)
        graph.create(link_meta)
        link_project = Relationship(project, 'PRODUCE', model)
        graph.create(link_project)
        celery_tasks.create_landmark_learning.delay(learn_data, annotation, model, strategy_model.to_dict())
        return model['path']
    except:
        model = matcher.match('Model', path=model['path']).first()
        if model['status'] != 'Created':
            celery_tasks.create_landmark_learning.delay(learn_data, annotation, model, strategy_model.to_dict())
            return model['path']
        return False


def learning_landmark(project_name, username, sample_name, body=None, region_name=None):  # noqa: E501
    """Create a landmark learning procedure

     # noqa: E501

    :param project_name: Name of specific project
    :type project_name: str
    :param username: Username
    :type username: str
    :param sample_name: Name of specific sample
    :type sample_name: str
    :param body:
    :type body: dict | bytes
    :param region_name: Name of specific Annotation Region of Interest
    :type region_name: str

    :rtype: None
    """
    if connexion.request.is_json:
        body = LearningStrategy.from_dict(connexion.request.get_json())  # noqa: E501
        status = db_create_landmark_learning(body, project_name, username, sample_name, region_name,
                                             'deep_learning/pose_estimation/hrnetv2_w18_imagenet_pretrained.pth')
        if status:
            return status, 201
    return 'do some magic!'
