def get_global_ci_id(ci: dict, tenant_id: str | None = None) -> str:
    """
    Get a global CI ID for a given CI object.
    The global CI ID is constructed as "{tenant_id}:{project_id}:{ci_id}".
    """
    _tenant_id = ci.get("tenant_id") or tenant_id
    _ci_id = ci.get("id") or ci.get("ci_id") or ""
    ci_global_id = (
        f"{_tenant_id}__"
        f"ci_{_ci_id}"
    )
    return str(ci_global_id)

def get_global_document_id(document_id: str, tenant_id: str | None = None, project_id: str | None = None) -> str:
    """
    Get a global document ID for a given document ID.
    The global document ID is constructed as "{tenant_id}:{project_id}:{document_id}".
    """
    global_document_id = (
        f"{tenant_id}__"
        f"{project_id}__"
        f"{document_id}"
    )
    return str(global_document_id)


def get_rls_file_s3_extraction_prefix(tenant_name: str, project_id: str, file_name: str) -> str:
    """
    Python port of the frontend's getRLSFileS3ExtractionPrefix().

    Top-level "extractions/" prefix so the BucketAV S3 event can be filtered
    to skip this path. file_name may or may not carry an extension (e.g.
    "temp.pdf" or "temp") — Path(...).stem handles both, mirroring Node's
    path.parse(fileName).name.
    """
    from pathlib import PurePosixPath
    file_base_name = PurePosixPath(file_name).stem
    return f"extractions/{tenant_name}/{project_id}/{file_base_name}"