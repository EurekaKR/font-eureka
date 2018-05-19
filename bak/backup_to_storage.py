import sys
from glob import glob

from boto3 import client
from botocore.exceptions import ClientError


class BackUp(object):
    bucket_name = 'bak'
    service_name = 's3'
    endpoint_url = 'http://kr.objectstorage.ncloud.com'
    region_name = 'kr-standard'

    def __init__(self, access_key, secret_key):
        self.__access_key = access_key
        self.__secret_key = secret_key

        self.s3 = client(
            self.service_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.__access_key,
            aws_secret_access_key=self.__secret_key
        )


    def upload(self, src_path: str, dest_path: str) -> None:
        '''Upload files in src_path to dest_path'''
        self.s3.upload_file(src_path, self.bucket_name, dest_path)


    def download(self, src_path: str, dest_path: str) -> None:
        '''Download files in src_path to dest_path'''
        self.s3.download_file(self.bucket_name, src_path, dest_path)


    def get_list(self):
        '''Print all object list in s3 bucket'''
        response = self.s3.list_objects(Bucket=self.bucket_name)

        print('list all in the bucket')

        for obj in response['Contents']:
            print('{0}\t{1}'.format(obj['Key'], obj['Size']))
