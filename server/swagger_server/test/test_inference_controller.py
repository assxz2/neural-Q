# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.rectangle import Rectangle  # noqa: E501
from swagger_server.test import BaseTestCase


class TestInferenceController(BaseTestCase):
    """InferenceController integration test stubs"""

    def test_inference_landmark(self):
        """Test case for inference_landmark

        Return a specific region inference
        """
        query_string = [('project_name', 'project_name_example'),
                        ('username', 'username_example'),
                        ('sample_name', 'sample_name_example'),
                        ('region_name', 'region_name_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/inference/landmark',
            method='POST',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_inference_landmark_status(self):
        """Test case for inference_landmark_status

        Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri
        """
        query_string = [('uri', 'uri_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/inference/landmark',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_inference_tracking(self):
        """Test case for inference_tracking

        Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri
        """
        body = Rectangle()
        query_string = [('project_name', 'project_name_example'),
                        ('username', 'username_example'),
                        ('sample_name', 'sample_name_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/inference/tracking',
            method='POST',
            data=json.dumps(body),
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_inference_tracking_status(self):
        """Test case for inference_tracking_status

        Inference a specific ROI sequence for submitted ROI and sample and return the resouces uri
        """
        query_string = [('uri', 'uri_example')]
        response = self.client.open(
            '/liammm-lza/rouran/v1/inference/tracking',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
