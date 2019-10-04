#!/usr/bin/env python
from qiskit import IBMQ
from qiskit.providers import BaseBackend, JobStatus
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend_name", type=str, default='ibmq_20_tokyo', help="backend name")
    args = parser.parse_args()
    IBMQ.load_accounts()
    backend = IBMQ.get_backend(args.backend_name)
    for j in backend.jobs():
        status = j.status()
        if status == JobStatus.ERROR:
            print(j.status(), "Error message: ", j.error_message())
        else:
            print(j.status(), "Queue position: ", j.queue_position())
