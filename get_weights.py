import os
import logging

from urllib.parse import urlparse
from azure.datalake.store import core, lib
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

BACKEND_TYPE = os.environ.get('BACKEND_TYPE')
WEIGHTS_LOCATION = os.environ.get('WEIGHTS_LOCATION')


class DataLakeFileSystem(core.AzureDLFileSystem):
    RESOURCE = 'https://datalake.azure.net/'
    STORE_NAME = os.environ.get('AZURE_NAME')
    TENANT_ID = os.environ.get('AZURE_TENANT_ID')
    CLIENT_ID = os.environ.get('AZURE_CLIENT_ID')
    CLIENT_SECRET = os.environ.get('AZURE_CLIENT_SECRET')

    def __init__(self):
        adl_credits = lib.auth(tenant_id=self.TENANT_ID, client_secret=self.CLIENT_SECRET,
                               client_id=self.CLIENT_ID, resource=self.RESOURCE)
        super().__init__(token=adl_credits, store_name=self.STORE_NAME)


class BlobFileSystem:
    # Configuration option 1:
    CONNECTION_STRING = os.environ.get('AZURE_BLOB_CONNECTION_STRING')
    CREDENTIALS = os.environ.get('AZURE_BLOB_CREDENTIALS')
    CONTAINER_NAME = os.environ.get('AZURE_BLOB_CONTAINER')  # Note: only lowercase supported!

    # Configuration option 2:
    SAS_URI = os.environ.get('AZURE_BLOB_SAS_URI')

    # Additional settings:
    PRESERVE_LEADING_SLASH = os.environ.get('AZURE_BLOB_PRESERVE_LEADING_SLASH') in ['1', 'YES', 'yes', 'ON', 'on']
    SUPPORTED_PERMISSIONS = ['r']

    def __init__(self):
        # Configuration option 2:
        if BlobFileSystem.SAS_URI:
            self.connection_string = 'BlobEndpoint={};'.format(BlobFileSystem.SAS_URI)
            self.container_name = urlparse(BlobFileSystem.SAS_URI).path
            self.container_name = self.container_name[
                                  self.container_name.rindex('/') + 1:]  # strip everything up to the last '/'
            self.credentials = None
            self.blob_service_client = None
            self.container_client = ContainerClient.from_container_url(BlobFileSystem.SAS_URI)

        # Configuration option 1:
        else:
            self.connection_string = BlobFileSystem.CONNECTION_STRING
            self.container_name = BlobFileSystem.CONTAINER_NAME
            self.credentials = BlobFileSystem.CREDENTIALS
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string,
                credential=self.credentials
            )
            self.container_client = self._get_container_client(self.container_name, ensure_creation=True)

    def _get_container_client(self, container_name, ensure_creation=False):
        if ensure_creation and not any(container_info['name'] == container_name for container_info in
                                       self.blob_service_client.list_containers(name_starts_with=container_name)):
            return self.blob_service_client.create_container(container_name)
        else:
            return self.blob_service_client.get_container_client(container_name)

    @classmethod
    def pre_process_path(cls, path):
        if not cls.PRESERVE_LEADING_SLASH and path.startswith('/'):
            return path[1:]
        else:
            return path

    def get(self, path, filename):
        path = self.pre_process_path(path)
        blob_client = self.container_client.get_blob_client(path)
        with open(filename, 'wb') as fh:
            fh.write(blob_client.download_blob().readall())


def get_weights():
    weights_dir = 'weights'
    weights_location = os.path.join(weights_dir, 'weights.h5')
    if os.path.isfile(weights_location):
        logging.info('Model weights file already available. No new file downloaded')
    else:
        if BACKEND_TYPE == 'DATALAKE':
            fs = DataLakeFileSystem()
        elif BACKEND_TYPE == 'BLOBSTORAGE':
            fs = BlobFileSystem()
        else:
            raise Exception('Backend not properly configured, set BACKEND_TYPE env var')

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        logging.info('Retrieving model weights.')
        fs.get(WEIGHTS_LOCATION, weights_location)
        logging.info('Finished retrieving weights.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    get_weights()
