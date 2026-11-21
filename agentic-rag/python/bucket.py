import sys
from typing import Dict, List

import boto3

S3_URL = "http://local-s3-service.ezdata-system.svc.cluster.local:30000"

#=========================================================================

def create_bucket(bucket_name: str) -> None:
    s3 = boto3.client("s3", endpoint_url=S3_URL)
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")

#=========================================================================

def list_bucket_objects(bucket_name: str) -> None:
    s3 = boto3.client("s3", endpoint_url=S3_URL)
    print(f"Listing objects in bucket '{bucket_name}':")
    paginator = s3.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                print(key)
                found = True

    if not found:
        print(f"No objects found in bucket '{bucket_name}'.")

#=========================================================================

def clear_bucket(bucket_name: str) -> int:
    s3 = boto3.client("s3", endpoint_url=S3_URL)

    total_deleted = 0
    key_marker = None
    version_id_marker = None

    while True:
        kwargs: Dict[str, str] = {"Bucket": bucket_name}
        if key_marker:
            kwargs["KeyMarker"] = key_marker
        if version_id_marker:
            kwargs["VersionIdMarker"] = version_id_marker

        response = s3.list_object_versions(**kwargs)

        objects_to_delete: List[Dict[str, str]] = []

        for v in response.get("Versions", []):
            objects_to_delete.append({"Key": v["Key"], "VersionId": v["VersionId"]})

        for dm in response.get("DeleteMarkers", []):
            objects_to_delete.append({"Key": dm["Key"], "VersionId": dm["VersionId"]})

        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            s3.delete_objects(Bucket=bucket_name, Delete={"Objects": batch})
            total_deleted += len(batch)

        if not response.get("IsTruncated"):
            break

        key_marker = response.get("NextKeyMarker")
        version_id_marker = response.get("NextVersionIdMarker")

    return total_deleted

#=========================================================================


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python main.py <bucket-name>")
        sys.exit(1)

    #--- Clear bucket
    #count = clear_bucket(sys.argv[1])
    #print(f"Deleted {count} object(s) from bucket '{sys.argv[1]}'.")

	#--- Create bucket
    #create_bucket(sys.argv[1])

    #--- List bucket
    #list_bucket_objects(sys.argv[1])
