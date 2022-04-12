# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.learning_strategy import LearningStrategy  # noqa: E501
from swagger_server.test import BaseTestCase


class TestLearningController(BaseTestCase):
    """LearningController integration test stubs"""

    def test_learning_landmark(self):
        """Test case for learning_landmark

        Create a landmark learning procedure
        """
        body = LearningStrategy()
        query_string = [('project_name', 'project_name_example'),
                        ('username', 'username_example'),
                        ('sample_name', 'sample_name_example'),
                        ('region_name', 'region_name_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/learning/landmark',
            method='POST',
            data=json.dumps(body),
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
