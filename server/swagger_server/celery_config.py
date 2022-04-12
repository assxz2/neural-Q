# Broker setting.
broker_url = 'pyamqp://rabbitmq:rabbitmq@rabbit:5672//'
BROKER_URL = 'pyamqp://rabbitmq:rabbitmq@rabbit:5672//'
# broker = 'amqp://rabbitmq:rabbitmq@rabbit:5672//'

# List of modules to import when the Celery worker starts.
imports = ('swagger_server.controllers.inference_controller',
           'swagger_server.controllers.learning_controller',
           'swagger_server.controllers.label_controller')

# Using the database to store task state and results.
result_backend = 'redis://redis:6379/0'
