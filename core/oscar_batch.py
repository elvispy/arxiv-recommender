import json
import subprocess
import os
import time
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
OSCAR_HOST = "oscar-campus"
REMOTE_DIR = "~/scratch/arxiv-recommender"

class OscarBatchManager:
    """Manages asynchronous batch offloading to OSCAR via sbatch and polling."""
    
    def __init__(self, host=OSCAR_HOST):
        self.host = host

    def run_remote_batch(self, papers, poll_interval=300):
        """
        1. scp scripts and data
        2. sbatch job
        3. poll squeue
        4. scp results back
        """
        if not papers:
            return []

        local_in = "batch_in.json"
        local_out = "batch_out.json"
        with open(local_in, 'w') as f:
            json.dump(papers, f)

        try:
            # 1. Setup Remote Directory Explicitly
            logger.info(f"Ensuring remote directory {REMOTE_DIR} exists on {self.host}...")
            subprocess.run(["ssh"] + SSH_OPTS + [self.host, f"mkdir -p {REMOTE_DIR}"], check=True)
            
            # 2. Bootstrap & Transfer Setup Script
            logger.info(f"Bootstrapping remote environment on {self.host}...")
            subprocess.run(["scp"] + SSH_OPTS + ["remote/remote_setup.sh", f"{self.host}:{REMOTE_DIR}/remote_setup.sh"], check=True)
            subprocess.run(["ssh"] + SSH_OPTS + [self.host, f"bash {REMOTE_DIR}/remote_setup.sh"], check=True)

            # 3. Transfer Job Files
            logger.info("Transferring batch data and scripts...")
            subprocess.run(["scp"] + SSH_OPTS + [local_in, f"{self.host}:{REMOTE_DIR}/batch_in.json"], check=True)
            subprocess.run(["scp"] + SSH_OPTS + ["remote/oscar_infer.py", f"{self.host}:{REMOTE_DIR}/oscar_infer.py"], check=True)
            subprocess.run(["scp"] + SSH_OPTS + ["remote/run_inference.sh", f"{self.host}:{REMOTE_DIR}/run_inference.sh"], check=True)

            # 3. Submit sbatch
            submit_cmd = f"cd {REMOTE_DIR} && sbatch run_inference.sh batch_in.json batch_out.json"
            result = subprocess.run(["ssh"] + SSH_OPTS + [self.host, submit_cmd], capture_output=True, text=True, check=True)
            
            # Extract Job ID: "Submitted batch job 1234567"
            match = re.search(r"job (\d+)", result.stdout)
            if not match:
                logger.error(f"Failed to parse job ID: {result.stdout}")
                return []
            
            job_id = match.group(1)
            logger.info(f"Job submitted to Slurm: {job_id}")

            # 4. Polling Loop
            while True:
                time.sleep(poll_interval)
                check_cmd = f"squeue -j {job_id} -h -o %t"
                status_res = subprocess.run(["ssh"] + SSH_OPTS + [self.host, check_cmd], capture_output=True, text=True)
                
                status = status_res.stdout.strip()
                if not status: # Job is gone from queue
                    logger.info(f"Job {job_id} finished (no longer in queue).")
                    break
                
                logger.info(f"Job {job_id} status: {status}. Waiting {poll_interval}s...")

            # 5. Retrieve Results
            logger.info("Retrieving results from OSCAR...")
            subprocess.run(["scp"] + SSH_OPTS + [f"{self.host}:{REMOTE_DIR}/batch_out.json", local_out], check=True)

            with open(local_out, 'r') as f:
                results = json.load(f)
            return results

        except subprocess.CalledProcessError as e:
            logger.error(f"OSCAR offload failed: {e}")
            return []
        finally:
            if os.path.exists(local_in): os.remove(local_in)
            if os.path.exists(local_out): os.remove(local_out)
