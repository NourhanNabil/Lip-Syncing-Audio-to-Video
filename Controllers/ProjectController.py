from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse

from Utils.FileUtils import FileUtils

lip_syncing_router = APIRouter()


@lip_syncing_router.post("/upload")
async def generate_lip_syncing(video: UploadFile = File(...), audio: UploadFile = File(...)):
    video_content = await video.read()
    audio_content = await audio.read()
    
    video_path = FileUtils.save_audio_video(video.filename,video_content)
    audio_path = FileUtils.save_audio_video(audio.filename,audio_content)

    try:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Lip-syncing video generated successfully"})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))