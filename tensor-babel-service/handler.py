import json

import astunparse

import boto3


def babel(event, context):

    print astunparse

    print boto3

    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
