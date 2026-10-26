"""HTTP endpoints for Koi's supervised runner."""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

app = FastAPI(title="Koi")
app.state.runner = None
app.state.shutting_down = False


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Report runner process liveness without querying external services."""
    runner = request.app.state.runner
    healthy = (
        runner is not None and runner.returncode is None and not request.app.state.shutting_down
    )
    return JSONResponse(
        content={"status": "ok" if healthy else "unhealthy"},
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.post("/job", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def submit_job() -> None:
    """Reserve job submission until its Store integration is implemented."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@app.get("/resource_map", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_resource_map() -> None:
    """Reserve resource-map access until its Store integration is implemented."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
