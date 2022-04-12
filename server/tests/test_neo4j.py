import datetime
import pathlib
import unittest

from passlib.hash import bcrypt
import neotime
from py2neo import Graph, NodeMatcher
from py2neo.data import Node, Relationship


class Neo4jTest(unittest.TestCase):
    def setUp(self) -> None:
        self.username = 'lccurious'
        self.graph = Graph('http://neo4j:7474/db/data', user='neo4j', password='neo4jtest2020')
        self.matcher = NodeMatcher(self.graph)

    def test_create_constraints(self):
        """
        Create some constraints for Graph Database logic protection.

        :return:
        """
        self.graph.schema.create_uniqueness_constraint('Sample', 'title')
        self.graph.schema.create_uniqueness_constraint('User', 'username')
        self.graph.schema.create_uniqueness_constraint('Taxonomy', 'title')
        self.graph.schema.create_uniqueness_constraint('Model', 'path')

    def test_create_taxonomy(self):
        """
        CREATE (p: Taxonomy {
            title: 'Drosophila/VNC',
            description: 'Drosophila',
            timestamp: 'now()'
        })

        :return:
        """
        node_label = 'Taxonomy'
        taxonomy_node = Node(node_label,
                             title='Drosophila/VNC',
                             description="ventral nerve cord, The ventral nerve cord (VNC) "
                                         "and CB are generated in the embryo from neuroblasts "
                                         "delaminating from the embryonic neuroepithelium.",
                             timestamp=str(neotime.DateTime.now()))
        taxonomy_root = Node(node_label,
                             title='Drosophila',
                             description="Drosophila (/drəˈsɒfɪlə, drɒ-, droʊ-/[1][2]) is a genus of flies, "
                                         "belonging to the family Drosophilidae, whose members are often called "
                                         "\"small fruit flies\" or (less frequently) pomace flies, vinegar flies, "
                                         "or wine flies, a reference to the characteristic of many species to "
                                         "linger around overripe or rotting fruit.",
                             timestamp=str(neotime.DateTime.now()))
        rel = Relationship(taxonomy_root, 'HAS_PART', taxonomy_node)
        self.graph.create(rel)

    def test_create_users(self):
        """
        CREATE (p: User {
            username: 'neo4j_test',
            password: 'test_password',
            name: 'Albert Einstein',
            email: 'einstein@example.com'
            timestamp: 'now()',
            avatar: url
        })

        :return:
        """
        user = Node('User',
                    username='wyj',
                    password=bcrypt.encrypt('wyj'),
                    name='syx',
                    email='einstein@example.com',
                    timestamp=str(neotime.DateTime.now())
                    )
        user_s = Node('User',
                      username='guoli',
                      password=bcrypt.encrypt('guoli'),
                      name='cjj',
                      email='sunyixuan@zju.edu.cn',
                      timestamp=str(neotime.DateTime.now())
                      )
        self.graph.create(user)
        self.graph.create(user_s)

    def test_create_samples(self):
        """
        Samples creation must connect to User Node and Taxonomy Node.
        And path must belong to a relative path which follow:
        `special/record_modality/body_part/filename`

        CREATE (p: Sample {
            title: 'VNC/test1.avi',
            frames: 1121
            creator: 'Sun Yixuan'
            tag: 'GAL-25001'
            taxonomy: 'Drosophila/VNC',
            description: 'An OK sample.',
            timestamp: 'now()',
        })

        :return:
        """
        node_label = 'Sample'
        sample1 = Node(node_label,
                       title="test_videos/syx/syx_2021-1-21_4-1-2.tif",
                       frame=300,
                       creator='guoli',
                       taxonomy='test_videos/guoli329/20201221155425-886.tif',
                       tag='GAL4',
                       description='An OK sample.',
                       timestamp=str(neotime.DateTime.now()))
        sample2 = Node(node_label,
                       title="test_videos/guoli401/2021323155259 (438-638) cst.tif",
                       frame=300,
                       creator='guoli',
                       taxonomy='drosophila/light-sheet/VNC',
                       tag='GAL4',
                       description='An OK sample.',
                       timestamp=str(neotime.DateTime.now()))
        sample3 = Node(node_label,
                       title="test_videos/guoli401/2021323155259(29-329) cst.tif",
                       frame=300,
                       creator='guoli',
                       taxonomy='Drosophila/VNC',
                       tag='GAL4',
                       description='An OK sample.',
                       timestamp=str(neotime.DateTime.now()))

        taxonomy = self.matcher.match('Taxonomy', title='Drosophila/VNC').first()
        creator = self.matcher.match('User', username='guoli').first()
        rel_class1 = Relationship(sample1, 'INSTANCE_OF', taxonomy)
        rel_class2 = Relationship(sample2, 'INSTANCE_OF', taxonomy)
        rel_class3 = Relationship(sample3, 'INSTANCE_OF', taxonomy)
        self.graph.create(rel_class1)
        #self.graph.create(rel_class2)
        #self.graph.create(rel_class3)
        rel_creator1 = Relationship(creator, 'RECORD', sample1)
        rel_creator2 = Relationship(creator, 'RECORD', sample2)
        rel_creator3 = Relationship(creator, 'RECORD', sample3)
        self.graph.create(rel_creator1)
        #self.graph.create(rel_creator2)
        #self.graph.create(rel_creator3)

    def test_create_meta_models(self):
        """
        CREATE (p: Model {
            title: 'hrnetv2_w18_imagenet_pretrained.pth',
            task: 'pose_estimation'/'tracking'/'super_resolution',
            path: 'deep_learning/pose_estimation/hrnetv2_w18_imagenet_pretrained.pth'
            creator: 'neo4j_test',
            iteration: 31,
            status: 'ready'/'training'/'initialized',
            timestamp: 'now()',
        })

        :return:
        """
        node_label = 'Model'
        model_pose = Node(node_label,
                          title='hrnetv2_w18_imagenet_pretrained.pth',
                          task='pose_estimation',
                          path='deep_learning/pose_estimation/hrnetv2_w18_imagenet_pretrained.pth',
                          creator=self.username,
                          iteration=2000,
                          status='ready',
                          timestamp=str(neotime.DateTime.now()))
        model_tracking = Node(node_label,
                              title='CIResNet22_RPN.pth',
                              task='tracking',
                              path='deep_learning/tracking/CIResNet22_RPN.pth',
                              creator=self.username,
                              iteration=2000,
                              status='ready',
                              timestamp=str(neotime.DateTime.now()))
        self.graph.create(model_pose)
        self.graph.create(model_tracking)

    def test_create_project(self):
        """
        Title are not allowed to use space.

        CREATE (p: Project {
            title: 'A-rouran-test-project',
            creator: 'neo4j_test',
            path: 'username/project_name'
            description: 'Some description',
            datetime: 'now()'
        })

        :return:
        """
        suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        project_path = pathlib.Path(self.username).joinpath('-'.join(['A-rouran-test-project', suffix]))
        project = Node('Project',
                       title='A-rouran-test-project',
                       creator=self.username,
                       path=str(project_path),
                       description='Some description here.',
                       timestamp=neotime.DateTime.now())

        link_samples = ["drosophila/light-sheet/VNC/20180829_4-1.avi",
                        "drosophila/light-sheet/VNC/48946_400-700.avi",
                        "drosophila/light-sheet/VNC/49939_2.avi"]

        user = self.matcher.match('User', username=self.username).first()
        u2project = Relationship(user, 'CREATE', project)
        self.graph.create(u2project)
        for sample in link_samples:
            sample_node = self.matcher.match('Sample', title=sample).first()
            rel_sample2project = Relationship(sample_node, 'IMPORT', project)
            self.graph.create(rel_sample2project)

    def test_create_annotation(self):
        """
        CREATE (p: Annotation {
            title: 'A Random name (hashed)',
            label: 'Points',
            creator: 'neo4j_test',
            sample: 'VNC/test1.avi',
            path: 'username/project_name/points_datetime.csv'
        })

        :return:
        """
        node_label = 'Annotation'
        project = self.matcher.match('Project', title='zz', creator='lza').first()
        suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        rect_path = '-'.join(['Rect', suffix])
        annotation_rect = Node(node_label,
                               title='VNC-ROI',
                               label='Rect',
                               creator=self.username,
                               sample='drosophila/light-sheet/VNC/test1.avi',
                               path='{u}/{p}/{a}'.format(u='lza', p=project['path'], a=rect_path),
                               timestamp=neotime.DateTime.now())
        suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        points_path = '-'.join(['Points', suffix])
        annotation_points = Node(node_label,
                                 title='A random Pose',
                                 label='Points',
                                 creator=self.username,
                                 sample='drosophila/light-sheet/VNC/test1.avi',
                                 path='{u}/{p}/{a}'.format(u=self.username, p=project['path'], a=points_path),
                                 timestamp=neotime.DateTime.now())

        sample1 = self.matcher.match('Sample', title='test_videos').first()
        rel_rect2sample = Relationship(annotation_rect, 'ANNOTATE', sample1)
        self.graph.create(rel_rect2sample)
        rel_points2sample = Relationship(annotation_points, 'ANNOTATE', sample1)
        self.graph.create(rel_points2sample)
        rel_p2rect = Relationship(project, 'BRING', annotation_rect)
        self.graph.create(rel_p2rect)
        rel_p2points = Relationship(project, 'BRING', annotation_points)
        self.graph.create(rel_p2points)

    def test_train_project(self):
        """
        CREATE (p: Model {
            name: 'A Random name (hashed)',
            creator: 'neo4j_test',
            train_data: ['points/training_dataset'],
            best_accuracy: 0.96,
            datetime: '2020-09-28 08:28:12',
            iteration: 31,
            batch_size: 32
        })

        :return:
        """
        model = Node('Model',
                     title='finetuned_hrnetv2',
                     creator=self.username,
                     status='initialized',
                     iteration=50,
                     timestamp=neotime.DateTime.now())
        project = self.matcher.match('Project', title='A-rouran-test-project', creator=self.username).first()
        annotation = self.matcher.match('Annotation', title='A random Pose', creator=self.username).first()
        sample1 = self.matcher.match('Sample', title='drosophila/light-sheet/VNC/test1.avi').first()
        base_model = self.matcher.match('Model', title='hrnetv2_w18_imagenet_pretrained.pth').first()

        sample2model = Relationship(sample1, 'FEED', model)
        self.graph.create(sample2model)
        annotation2model = Relationship(annotation, 'FEED', model)
        self.graph.create(annotation2model)
        p2model = Relationship(project, 'PRODUCE', model)
        self.graph.create(p2model)
        rel_fine_tune = Relationship(base_model, 'FINE_TUNE', model)
        self.graph.create(rel_fine_tune)

    def test_evaluate_project(self):
        """
        CREATE (p: Result {
            title='result name hash',
            label='Points',
            timestamp='now()'
        })
        CREATE (p: Query {
            title='get ROI',
            creator='neo4j_test',
            label='Rect',
            path='',
            detail=''
        })

        :return:
        """
        result_roi = Node('Result',
                          title='ROI-test',
                          label='Rect',
                          timestamp=neotime.DateTime.now())
        query_roi = Node('Query',
                         title='GET ROI',
                         creator=self.username,
                         label='Rect',
                         path='',
                         detail='[255.7, 128.5, 480, 360]')

        result_points = Node('Result',
                             title='Points-test',
                             label='Points',
                             timestamp=neotime.DateTime.now())
        query_points = Node('Query',
                            title='GET POINTS',
                            creator=self.username,
                            label='Points',
                            path='',
                            detail='')

        model_roi = self.matcher.match('Model', title='CIResNet22_RPN.pth', creator=self.username).first()
        rel_m = Relationship(model_roi, 'INFERENCE', result_roi)
        self.graph.create(rel_m)
        sample = self.matcher.match('Sample', title='drosophila/light-sheet/VNC/test1.avi').first()
        rel_s = Relationship(sample, 'HAS_INFO', result_roi)
        self.graph.create(rel_s)
        rel_query = Relationship(query_roi, 'ASK_FOR', result_roi)
        self.graph.create(rel_query)

        model_points = self.matcher.match('Model', title='finetuned_hrnetv2', creator=self.username).first()
        rel_m = Relationship(model_points, 'INFERENCE', result_points)
        self.graph.create(rel_m)
        rel_s = Relationship(sample, 'HAS_INFO', result_points)
        self.graph.create(rel_s)
        rel_query = Relationship(query_points, 'ASK_FOR', result_points)
        self.graph.create(rel_query)
