# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestQueueController(BaseTestCase):
    """QueueController integration test stubs"""

    def test_queue_task_id_get(self):
        """Test case for queue_task_id_get

        Get task information in queue
        """
        response = self.client.open(
            '/liammm-lza/rouran/v1/queue/{taskId}'.format(task_id='task_id_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
