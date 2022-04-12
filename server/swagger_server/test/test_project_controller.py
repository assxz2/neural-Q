# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.file_tree import FileTree  # noqa: E501
from swagger_server.models.inline_response201 import InlineResponse201  # noqa: E501
from swagger_server.models.project import Project  # noqa: E501
from swagger_server.test import BaseTestCase


class TestProjectController(BaseTestCase):
    """ProjectController integration test stubs"""

    def test_create_project(self):
        """Test case for create_project

        Create a project by given parameters
        """
        body = Project()
        response = self.client.open(
            '/liammm-lza/rouran/v1/project',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_download_file(self):
        """Test case for download_file

        Download file
        """
        query_string = [('path', 'path_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/project/downloadFile',
            method='POST',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_file_tree_show(self):
        """Test case for file_tree_show

        get file tree from current dir
        """
        query_string = [('location', 'location_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/project/fileTree',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
