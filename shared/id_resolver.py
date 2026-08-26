def get_global_ci_id(ci: dict, tenant_id: str | None = None, project_id: str | None = None) -> str:
    """
    Get a global CI ID for a given CI object.
    The global CI ID is constructed as "{tenant_id}:{project_id}:{ci_id}".
    """
    _tenant_id = ci.get("tenant_id") or tenant_id
    _project_id = ci.get("project_id") or project_id
    _ci_id = ci.get("id") or ci.get("ci_id") or ""
    ci_global_id = (
        f"{_tenant_id}__"
        f"{_project_id}__"
        f"ci_{_ci_id}"
    )
    return ci_global_id

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
    return global_document_id