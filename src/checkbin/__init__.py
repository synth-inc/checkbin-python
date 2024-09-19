# SPDX-FileCopyrightText: 2024-present Synth Inc
#
# SPDX-License-Identifier: MIT

import os
import time
import uuid
import json
import random
import pickle
import tempfile
from typing import Optional, Any, Tuple, Literal

import cv2
import numpy
import torch
import requests
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient


MediaType = Literal["image", "video"]


AUTH_TOKEN = None


def authenticate(token: str):
    global AUTH_TOKEN
    AUTH_TOKEN = token


class CheckbinRunner:
    def __init__(
        self,
        run_id: str,
        parent_id: str,
        input_state: Optional[Any] = None,
        azure_account_name: Optional[str] = None,
        azure_account_key: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        gcp_service_account_info: Optional[dict] = None,
        gcp_service_account_json: Optional[str] = None,
    ):
        self.run_id = run_id
        self.parent_id = parent_id
        self.input_state = input_state
        self.azure_account_name = azure_account_name
        self.azure_account_key = azure_account_key
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.gcp_service_account_info = gcp_service_account_info
        self.gcp_service_account_json = gcp_service_account_json
        self.checkins = []

    def checkpoint(
        self,
        name: str,
        ids: Optional[dict[str, str | int | float]] = None,
    ):
        self.checkins.append(
            {
                "name": name,
                "ids": ids,
                "state": None,
                "files": None,
            }
        )

    def add_state(self, key: str, state: Any):
        if self.checkins[-1]["state"] is None:
            self.checkins[-1]["state"] = {}
        self.checkins[-1]["state"][key] = state

    def add_file(
        self,
        key: str,
        url: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        if self.checkins[-1]["files"] is None:
            self.checkins[-1]["files"] = {}
        self.checkins[-1]["files"][key] = {
            "url": url,
            "mediaType": media_type,
            "pickle": pickle,
        }

    def check_credentials_azure(self):
        if self.azure_account_name is None:
            raise Exception("Azure account name is required")
        if self.azure_account_key is None:
            raise Exception("Azure account key is required")

    def upload_file_azure(
        self,
        container: str,
        key: str,
        file_path: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.check_credentials_azure()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{self.azure_account_name}.blob.core.windows.net",
            credential=self.azure_account_key,
        )
        container_client = blob_service_client.get_container_client(container)
        _, extension = os.path.splitext(os.path.basename(file_path))
        filename = f"{uuid.uuid4().hex}{extension}"
        blob_client = container_client.get_blob_client(filename)
        with open(file_path, "rb") as file:
            start_time = time.time()
            print(f"Checkbin: recording file")
            blob_client.upload_blob(file)
            print(f"Checkbin: recording file upload time: {time.time() - start_time}")
            print(f"Checkbin: recorded file: {blob_client.url}")
            url = blob_client.url
        self.add_file(key, url, media_type, pickle)

    def upload_pickle_azure(self, container_name: str, key: str, variable: Any):
        self.check_credentials_azure()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pickle.dump(variable, tmp_file)
            self.upload_file_azure(container_name, key, tmp_file.name, pickle=True)

    def check_credentials_aws(self):
        if self.aws_access_key is None:
            raise Exception("AWS access key is required")
        if self.aws_secret_key is None:
            raise Exception("AWS secret key is required")

    def upload_file_aws(
        self,
        bucket: str,
        key: str,
        file_path: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.check_credentials_aws()
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )
        _, extension = os.path.splitext(os.path.basename(file_path))
        filename = f"{uuid.uuid4().hex}{extension}"
        with open(file_path, "rb") as file:
            start_time = time.time()
            print(f"Checkbin: recording file")
            s3_client.upload_fileobj(file, bucket, filename)
            print(f"Checkbin: recording file upload time: {time.time() - start_time}")
            url = f"https://{bucket}.s3.amazonaws.com/{filename}"
            print(f"Checkbin: recorded file: {url}")
        self.add_file(key, url, media_type, pickle)

    def upload_pickle_aws(self, bucket: str, key: str, variable: Any):
        self.check_credentials_aws()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pickle.dump(variable, tmp_file)
            self.upload_file_aws(bucket, key, tmp_file.name, pickle=True)

    def check_credentials_gcp(self):
        if (
            self.gcp_service_account_info is None
            and self.gcp_service_account_json is None
        ):
            raise Exception("GCP service account info or json is required")

    def upload_file_gcp(
        self,
        bucket: str,
        key: str,
        file_path: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.check_credentials_gcp()
        if self.gcp_service_account_info is not None:
            storage_client = storage.Client.from_service_account_info(
                self.gcp_service_account_info
            )
        else:
            storage_client = storage.Client.from_service_account_json(
                self.gcp_service_account_json
            )
        bucket_client = storage_client.get_bucket(bucket)
        _, extension = os.path.splitext(os.path.basename(file_path))
        filename = f"{uuid.uuid4().hex}{extension}"
        blob_client = bucket_client.blob(filename)
        with open(file_path, "rb") as file:
            start_time = time.time()
            print(f"Checkbin: recording file")
            blob_client.upload_from_file(file)
            print(f"Checkbin: recording file upload time: {time.time() - start_time}")
            url = f"https://storage.googleapis.com/{bucket}/{filename}"
            print(f"Checkbin: recorded file: {url}")
        self.add_file(key, url, media_type, pickle)

    def upload_pickle_gcp(self, bucket: str, key: str, variable: Any):
        self.check_credentials_gcp()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pickle.dump(variable, tmp_file)
            self.upload_file_gcp(bucket, key, tmp_file.name, pickle=True)

    def colorspace_to_conversion(self, colorspace: str) -> Optional[int]:
        if colorspace == "BGRA":
            return cv2.COLOR_BGRA2BGR
        elif colorspace == "RGBA":
            return cv2.COLOR_RGBA2BGR
        elif colorspace == "RGB":
            return cv2.COLOR_RGB2BGR
        elif colorspace == "GRAY":
            return cv2.COLOR_GRAY2BGR
        elif colorspace == "YCrCb":
            return cv2.COLOR_YCrCb2BGR
        elif colorspace == "HSV":
            return cv2.COLOR_HSV2BGR
        elif colorspace == "Lab":
            return cv2.COLOR_Lab2BGR
        elif colorspace == "Luv":
            return cv2.COLOR_Luv2BGR
        elif colorspace == "HLS":
            return cv2.COLOR_HLS2BGR
        elif colorspace == "YUV":
            return cv2.COLOR_YUV2BGR
        else:
            return None

    def upload_array_as_image(
        self,
        container: str,
        storage_service: Literal["azure", "aws", "gcp"],
        key: str,
        array: numpy.ndarray | torch.Tensor,
        range: Tuple[int, int] = (0, 255),
        colorspace: Optional[
            Literal[
                "BGR",
                "BGRA",
                "RGBA",
                "RGB",
                "GRAY",
                "YCrCb",
                "HSV",
                "Lab",
                "Luv",
                "HLS",
                "YUV",
            ]
        ] = None,
    ):
        if storage_service == "azure":
            self.check_credentials_azure()
        elif storage_service == "aws":
            self.check_credentials_aws()
        elif storage_service == "gcp":
            self.check_credentials_gcp()

        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()

        min_val, max_val = range
        array = ((array - min_val) * 255) / (max_val - min_val)

        if (
            colorspace is not None
            and self.colorspace_to_conversion(colorspace) is not None
        ):
            array = cv2.cvtColor(array, self.colorspace_to_conversion(colorspace))

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            cv2.imwrite(tmp_file.name, array)
            if storage_service == "azure":
                self.upload_file_azure(container, key, tmp_file.name, "image")
            elif storage_service == "aws":
                self.upload_file_aws(container, key, tmp_file.name, "image")
            elif storage_service == "gcp":
                self.upload_file_gcp(container, key, tmp_file.name, "image")

    def submit_test(self):
        self.checkpoint("Output")
        self.checkins[-1]["state"] = self.checkins[-2]["state"]
        self.checkins[-1]["files"] = self.checkins[-2]["files"]

        requests.post(
            "https://checkbin-server-prod-d332d31d3c50.herokuapp.com/checkin",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "runId": self.run_id,
                "parentId": self.parent_id,
                "checkins": self.checkins,
            },
            timeout=30,
        )


class CheckbinApp:
    def __init__(self, app_key: str):
        self.app_key = app_key
        self.azure_account_name = None
        self.azure_account_key = None
        self.aws_access_key = None
        self.aws_secret_key = None
        self.gcp_service_account_info = None
        self.gcp_service_account_json = None

    def add_azure_credentials(self, account_name: str, account_key: str):
        self.azure_account_name = account_name
        self.azure_account_key = account_key

    def add_aws_credentials(self, access_key: str, secret_key: str):
        self.aws_access_key = access_key
        self.aws_secret_key = secret_key

    def add_gcp_credentials_info(self, service_account_info: dict):
        self.gcp_service_account_info = service_account_info

    def add_gcp_credentials_json(self, service_account_json: str):
        self.gcp_service_account_json = service_account_json

    def start_run(
        self,
        checkin_id: Optional[str] = None,
        set_id: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> list[CheckbinRunner]:
        run_response = requests.post(
            "https://checkbin-server-prod-d332d31d3c50.herokuapp.com/run",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"appKey": self.app_key},
            timeout=30,
        )
        run_data = json.loads(run_response.content)
        run_id = run_data["id"]

        checkins = []
        if checkin_id is not None:
            checkin_response = requests.get(
                f"https://checkbin-server-prod-d332d31d3c50.herokuapp.com/checkin/{checkin_id}",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                params={"includeState": "true"},
                timeout=30,
            )
            checkin = json.loads(checkin_response.content)
            checkins = [checkin]
        elif set_id is not None:
            set_response = requests.get(
                f"https://checkbin-server-prod-d332d31d3c50.herokuapp.com/set/{set_id}",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                params={"includeCheckins": "true", "includeState": "true"},
                timeout=30,
            )
            set = json.loads(set_response.content)
            checkins = set["checkins"]
            if sample_size is not None:
                checkins = random.sample(checkins, min(sample_size, len(checkins)))
        else:
            raise Exception("Either checkin_id or set_id must be provided")

        runners = []
        for checkin in checkins:
            runner = CheckbinRunner(
                run_id=run_id,
                parent_id=checkin["id"],
                input_state={state["name"]: state for state in checkin["state"]},
                azure_account_name=self.azure_account_name,
                azure_account_key=self.azure_account_key,
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                gcp_service_account_info=self.gcp_service_account_info,
                gcp_service_account_json=self.gcp_service_account_json,
            )
            runners.append(runner)
        return runners
