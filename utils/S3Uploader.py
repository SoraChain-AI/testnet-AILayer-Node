import boto3
import botocore
import os

class S3Uploader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key)
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key


    def upload_config_folder(self, config_folder_path):
        # Get the list of files and directories in the config folder
        files_and_dirs = os.listdir(config_folder_path)
        
        # Iterate over the directory and its subdirectories
        for root, dirs, files in os.walk(config_folder_path):
            for file in files:
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                print(file_path)
                
                # Construct the key for the file in the S3 bucket
                key = os.path.relpath(file_path, config_folder_path).replace(os.sep, '/')

                # Upload the file
                self.s3.upload_file(file_path,Bucket=self.bucket_name ,Key=key)

    # def fetch_config_folder(self, config_dir_path):
    def fetch_config_folder(self,bucket_name, s3_folder, local_dir=None):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        s3_resource = boto3.resource('s3', aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key)
        bucket = s3_resource.Bucket(bucket_name)
        try:
            for obj in bucket.objects.filter(Prefix=s3_folder):
                target = obj.key if local_dir is None \
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == '/':
                    continue
                bucket.download_file(obj.key, target)
                print(f"Downloaded {obj.key} to {target}")
                os.chmod(target, 0o755)


        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
                
# Example usage:
# if __name__ == '__main__':
#     uploader = S3Uploader(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME)
#     config_folder_path = '/path/to/config/folder'
#     uploader.upload_config_folder(config_folder_path)