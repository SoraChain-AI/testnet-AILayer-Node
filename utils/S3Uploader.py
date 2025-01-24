import boto3
import os

class S3Uploader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key)
        self.bucket_name = bucket_name

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

# Example usage:
# if __name__ == '__main__':
#     uploader = S3Uploader(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME)
#     config_folder_path = '/path/to/config/folder'
#     uploader.upload_config_folder(config_folder_path)