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


class CheckbinFileUploader:
    def __init__(self):
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

    def check_credentials(self, storage_service: Literal["azure", "aws", "gcp"]):
        if storage_service == "azure":
            if self.azure_account_name is None:
                raise Exception("Azure account name is required")
            if self.azure_account_key is None:
                raise Exception("Azure account key is required")
        elif storage_service == "aws":
            if self.aws_access_key is None:
                raise Exception("AWS access key is required")
            if self.aws_secret_key is None:
                raise Exception("AWS secret key is required")
        elif storage_service == "gcp":
            if (
                self.gcp_service_account_info is None
                and self.gcp_service_account_json is None
            ):
                raise Exception("GCP service account info or json is required")

    def upload_file_azure(self, container: str, extension: str, file: bytes):
        blob_service_client = BlobServiceClient(
            account_url=f"https://{self.azure_account_name}.blob.core.windows.net",
            credential=self.azure_account_key,
        )
        container_client = blob_service_client.get_container_client(container)
        filename = f"{uuid.uuid4().hex}{extension}"
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(file)
        return blob_client.url

    def upload_file_aws(self, bucket: str, extension: str, file: bytes):
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )
        filename = f"{uuid.uuid4().hex}{extension}"
        s3_client.upload_fileobj(file, bucket, filename)
        return f"https://{bucket}.s3.amazonaws.com/{filename}"

    def upload_file_gcp(self, bucket: str, extension: str, file: bytes):
        if self.gcp_service_account_info is not None:
            storage_client = storage.Client.from_service_account_info(
                self.gcp_service_account_info
            )
        else:
            storage_client = storage.Client.from_service_account_json(
                self.gcp_service_account_json
            )
        bucket_client = storage_client.get_bucket(bucket)
        filename = f"{uuid.uuid4().hex}{extension}"
        blob_client = bucket_client.blob(filename)
        blob_client.upload_from_file(file)
        return f"https://storage.googleapis.com/{bucket}/{filename}"


class CheckbinCheckin:
    def __init__(
        self,
        file_uploader: CheckbinFileUploader,
        name: str,
        ids: Optional[dict[str, str | int | float]] = None,
    ):
        self.file_uploader = file_uploader
        self.name = name
        self.ids = ids
        self.keys = set()
        self.state = None
        self.files = None

    def get_state(self):
        return {
            "name": self.name,
            "ids": self.ids,
            "state": self.state,
            "files": self.files,
        }

    def __get_key(self, key: str) -> str:
        if key not in self.keys:
            return key

        level = 1
        new_key = key
        while new_key in self.keys:
            new_key = f"{key}_{level}"
            level += 1
        return new_key

    def add_state(self, key: str, state: Any):
        if self.state is None:
            self.state = {}
        key = self.__get_key(key)
        self.state[key] = state
        self.keys.add(key)

    def add_file(
        self,
        key: str,
        url: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        if self.files is None:
            self.files = {}
        key = self.__get_key(key)
        self.files[key] = {
            "url": url,
            "mediaType": media_type,
            "pickle": pickle,
        }
        self.keys.add(key)

    def upload_file(
        self,
        container: str,
        storage_service: Literal["azure", "aws", "gcp"],
        key: str,
        file_path: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.file_uploader.check_credentials(storage_service)

        with open(file_path, "rb") as file:
            start_time = time.time()
            print(f"Checkbin: recording file")
            _, extension = os.path.splitext(os.path.basename(file_path))
            if storage_service == "azure":
                url = self.file_uploader.upload_file_azure(container, extension, file)
            elif storage_service == "aws":
                url = self.file_uploader.upload_file_aws(container, extension, file)
            elif storage_service == "gcp":
                url = self.file_uploader.upload_file_gcp(container, extension, file)
            print(f"Checkbin: recording file upload time: {time.time() - start_time}")
            print(f"Checkbin: recorded file: {url}")

        self.add_file(key, url, media_type, pickle)

    def upload_pickle(
        self,
        container: str,
        storage_service: Literal["azure", "aws", "gcp"],
        key: str,
        variable: Any,
    ):
        self.file_uploader.check_credentials(storage_service)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pickle.dump(variable, tmp_file)
            self.upload_file(
                container, storage_service, key, tmp_file.name, pickle=True
            )

    def __colorspace_to_conversion(self, colorspace: str) -> Optional[int]:
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
        self.file_uploader.check_credentials(storage_service)

        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()

        min_val, max_val = range
        array = ((array - min_val) * 255) / (max_val - min_val)

        if (
            colorspace is not None
            and self.__colorspace_to_conversion(colorspace) is not None
        ):
            array = cv2.cvtColor(array, self.__colorspace_to_conversion(colorspace))

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            cv2.imwrite(tmp_file.name, array)
            self.upload_file(container, storage_service, key, tmp_file.name, "image")


class CheckbinRunner:
    def __init__(
        self,
        run_id: str,
        parent_id: str,
        file_uploader: CheckbinFileUploader,
        input_state: Optional[Any] = None,
    ):
        self.run_id = run_id
        self.parent_id = parent_id
        self.file_uploader = file_uploader
        self.input_state = input_state
        self.checkins: list[CheckbinCheckin] = []
        self.is_running = False

    def checkpoint(
        self,
        name: str,
        ids: Optional[dict[str, str | int | float]] = None,
    ):
        self.checkins.append(CheckbinCheckin(self.file_uploader, name, ids))

        if not self.is_running:
            requests.patch(
                f"https://checkbin-server-prod-d332d31d3c50.herokuapp.com/run/{self.run_id}/job",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                json={"jobs": [{"checkinId": self.parent_id, "status": "running"}]},
                timeout=30,
            )
            self.is_running = True

    def add_state(self, key: str, state: Any):
        self.checkins[-1].add_state(key, state)

    def add_file(
        self,
        key: str,
        url: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.checkins[-1].add_file(key, url, media_type, pickle)

    def upload_file(
        self,
        container: str,
        storage_service: Literal["azure", "aws", "gcp"],
        key: str,
        file_path: str,
        media_type: Optional[MediaType] = None,
        pickle: bool = False,
    ):
        self.checkins[-1].upload_file(
            container, storage_service, key, file_path, media_type, pickle
        )

    def upload_pickle(
        self,
        container: str,
        storage_service: Literal["azure", "aws", "gcp"],
        key: str,
        variable: Any,
    ):
        self.checkins[-1].upload_pickle(container, storage_service, key, variable)

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
        self.checkins[-1].upload_array_as_image(
            container, storage_service, key, array, range, colorspace
        )

    def submit_test(self):
        self.checkpoint("Output")
        self.checkins[-1].state = self.checkins[-2].state
        self.checkins[-1].files = self.checkins[-2].files

        requests.post(
            "https://checkbin-server-prod-d332d31d3c50.herokuapp.com/checkin",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "runId": self.run_id,
                "parentId": self.parent_id,
                "checkins": [checkin.get_state() for checkin in self.checkins],
            },
            timeout=30,
        )

        requests.patch(
            f"https://checkbin-server-prod-d332d31d3c50.herokuapp.com/run/{self.run_id}/job",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"jobs": [{"checkinId": self.parent_id, "status": "completed"}]},
            timeout=30,
        )


class CheckbinInputSet:
    def __init__(self, app_key: str, file_uploader: CheckbinFileUploader, name: str):
        self.app_key = app_key
        self.file_uploader = file_uploader
        self.name = name
        self.checkins: list[CheckbinCheckin] = []
        self.set_id = None

    def add_input(self):
        checkin = CheckbinCheckin(self.file_uploader, "Input")
        self.checkins.append(checkin)
        return checkin

    def submit_set(self):
        set_response = requests.post(
            "https://checkbin-server-prod-d332d31d3c50.herokuapp.com/set",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "appKey": self.app_key,
                "name": self.name,
                "isInput": True,
                "checkins": [checkin.get_state() for checkin in self.checkins],
            },
            timeout=30,
        )
        set_data = json.loads(set_response.content)
        self.set_id = set_data["id"]
        return self.set_id


class CheckbinApp:
    def __init__(self, app_key: str):
        self.app_key = app_key
        self.file_uploader = CheckbinFileUploader()

    def add_azure_credentials(self, account_name: str, account_key: str):
        self.file_uploader.add_azure_credentials(account_name, account_key)

    def add_aws_credentials(self, access_key: str, secret_key: str):
        self.file_uploader.add_aws_credentials(access_key, secret_key)

    def add_gcp_credentials_info(self, service_account_info: dict):
        self.file_uploader.add_gcp_credentials_info(service_account_info)

    def add_gcp_credentials_json(self, service_account_json: str):
        self.file_uploader.add_gcp_credentials_json(service_account_json)

    def create_input_set(self, name: str) -> CheckbinInputSet:
        return CheckbinInputSet(self.app_key, self.file_uploader, name)

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

        requests.patch(
            f"https://checkbin-server-prod-d332d31d3c50.herokuapp.com/run/{run_id}/job",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"jobs": [{"checkinId": checkin["id"]} for checkin in checkins]},
            timeout=30,
        )

        print(f"Checkbin: started run {run_id} with {len(checkins)} jobs")

        runners = []
        for checkin in checkins:
            runner = CheckbinRunner(
                run_id=run_id,
                parent_id=checkin["id"],
                file_uploader=self.file_uploader,
                input_state={state["name"]: state for state in checkin["state"]},
            )
            runners.append(runner)
        return runners
