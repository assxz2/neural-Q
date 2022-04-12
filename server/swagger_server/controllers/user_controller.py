import os

import connexion
import six

import neotime
from passlib.hash import bcrypt
from py2neo import Node
from swagger_server.models.user import User  # noqa: E501
from swagger_server import util, db_connection


def db_user_find(username):
    """Return the user node from neo4j query

    :param username: Username
    :type username: str

    :return: The User Node of this User
    :rtype: Node
    """
    graph, matcher = db_connection()
    user = matcher.match('User').where(f"_.username='{username}'").first()
    return user


def db_user_register(user_model):
    """Register user and insert to neo4j database

    :param user_model: a user class
    :type user_: User

    :return: If create user
    :rtype: bool
    """
    graph, matcher = db_connection()
    if db_user_find(user_model.username) is None:
        user = Node('User',
                    username=user_model.username,
                    password=bcrypt.encrypt(user_model.password),
                    name=user_model.first_name + ' ' + user_model.last_name,
                    email=user_model.email,
                    phone=user_model.phone,
                    status=user_model.user_status,
                    timestamp=str(neotime.DateTime.now()))
        graph.create(user)
        return True
    else:
        return False


def db_user_verify_password(username, password):
    """Verify the user password

    :param username: Username
    :type username: str
    :param password: Password
    :type: str
    :return: If password is correct
    :rtype: bool
    """
    user = db_user_find(username)
    if user is not None:
        return bcrypt.verify(password, user['password'])
    else:
        return False


def create_user(body):  # noqa: E501
    """Create user

    This can only be done by the logged in user. # noqa: E501

    :param body: Created user object
    :type body: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        body = User.from_dict(connexion.request.get_json())  # noqa: E501
        if db_user_register(body):
            return 'OK', 201
        elif db_user_find(body.username):
            return 'Username already existed', 202
        else:
            return 'Failed', 400


def delete_user(username):  # noqa: E501
    """Delete user

    This can only be done by the logged in user. # noqa: E501

    :param username: The name that needs to be deleted
    :type username: str

    :rtype: None
    """
    return 'do some magic!'


def get_user_by_name(username):  # noqa: E501
    """Get user by user name

     # noqa: E501

    :param username: The name that needs to be fetched. Use user1 for testing.
    :type username: str

    :rtype: User
    """
    return 'do some magic!'


def login_user(username, password):  # noqa: E501
    """Logs user into the system

     # noqa: E501

    :param username: The user name for login
    :type username: str
    :param password: The password for login in clear text
    :type password: str

    :rtype: str
    """
    if db_user_verify_password(username, password):
        return 'login success', 200
    elif db_user_find(username):
        return 'Password Error', 400
    else:
        return 'User not exists', 400


def logout_user():  # noqa: E501
    """Logs out current logged in user session

     # noqa: E501


    :rtype: None
    """

    return 'user logout'


def update_user(body, username):  # noqa: E501
    """Updated user

    This can only be done by the logged in user. # noqa: E501

    :param body: Updated user object
    :type body: dict | bytes
    :param username: name that need to be updated
    :type username: str

    :rtype: None
    """
    if connexion.request.is_json:
        body = User.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'