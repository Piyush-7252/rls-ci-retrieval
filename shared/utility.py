def getTenantFromEvent(event: dict) -> dict:
    """
    Extracts the tenant from the event dictionary.

    Args:
        event (dict): The event dictionary containing tenant information.

    Returns:
        dict: A dictionary containing tenant information with keys "tenant", "tenant_id", and "tenant_schema".
    """
    tenant = event.get("tenant", "")
    tenantName = tenant.get("name", "") if isinstance(tenant, dict) else ""
    tenantId = tenant.get("id", "") if isinstance(tenant, dict) else ""
    tenantSchema = tenant.get("schema", "") if isinstance(tenant, dict) else ""
    return {
        "name": tenantName,
        "id": tenantId,
        "schema": tenantSchema
    }