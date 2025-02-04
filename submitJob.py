
import argparse
import os

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper
from nvflare.fuel.hci.client.fl_admin_api_runner import TargetType
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.tool.job.job_cli import internal_submit_job, find_admin_user_and_dir
 

def main():
    args = define_parser()
    job_dir = args.job
    
    # admin_username, admin_user_dir = find_admin_user_and_dir()
    # print("admin_username, admin_user_dir", admin_username, admin_user_dir)

    # admin_username, admin_user_dir = find_admin_user_and_dir()


    try:
        internal_submit_job(args.admin_dir, args.username, args.job)
        # sess = new_secure_session(
        #     admin_username,
        #     admin_user_dir
        # )
        # print(sess.api.check_session_status_on_server())
        # job_id = sess.submit_job(job_dir)
        # print(job_id + " was submitted")
        # monitor_job() # waits until the job is done, see the section about it below for details
        # sess.monitor_job(job_id)
        print("job done!")
    finally:
        pass
        # sess.close()

def define_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--job", type=str, default=f"./workspace/SoraWorkspace/jobs/llm_hf_peft", help="Path to job config.")
    parser.add_argument("--admin_dir", type=str, default="${PWD}/workspace/SoraWorkspace/example_project/prod_00/admin@nvidia.com", help="Path to job config.")
    parser.add_argument("--username", type=str, default=f"admin@nvidia.com", help="Path to job config.")
    return parser.parse_args()

if __name__ == "__main__":
    main()