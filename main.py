from fastapi import FastAPI
import uvicorn

from Controllers import ProjectController
from Controllers import HealthController

from Helpers.Config import get_Settings

if __name__ == "__main__":

    app = FastAPI()
    app.include_router(ProjectController.lip_syncing_router)
    app.include_router(HealthController.health_router)
    uvicorn.run(app,host=get_Settings().SERVICE_HOST,port=get_Settings().SERVICE_PORT)