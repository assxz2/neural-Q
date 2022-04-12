# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.rectangle import Rectangle  # noqa: E501
from swagger_server.test import BaseTestCase


class TestLabelController(BaseTestCase):
    """LabelController integration test stubs"""

    def test_submit_rectangle(self):
        """Test case for submit_rectangle

        Add the annotation to frame
        """
        body = Rectangle()
        query_string = [('project_name', 'project_name_example'),
                        ('user_name', 'user_name_example'),
                        ('sample_name', 'sample_name_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/label/rectangle',
            method='POST',
            data=json.dumps(body),
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_submit_skeleton(self):
        """Test case for submit_skeleton

        Add the skeletons collection file to sample or sample clip
        """
        query_string = [('project_name', 'project_name_example'),
                        ('user_name', 'user_name_example'),
                        ('sample_name', 'sample_name_example'),
                        ('region_name', 'region_name_example')]
        data = dict(file_name='file_name_example')
        response = self.client.open(
            '/liammm-lza/rouran/v1/label/skeleton',
            method='POST',
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_skeleton(self):
        """Test case for update_skeleton

        Add the annotation to frame
        """
        query_string = [('project_name', 'project_name_example'),
                        ('user_name', 'user_name_example'),
                        ('sample_name', 'sample_name_example'),
                        ('region_name', 'region_name_example')]
        data = dict(file_name='file_name_example')
        response = self.client.open(
            '/liammm-lza/rouran/v1/label/skeleton',
            method='PUT',
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
