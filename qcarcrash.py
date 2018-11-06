from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
    imagenet_stats
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

cat_images_path = Path("/tmp")


model1 = {
    'name':'isDamaged',
    'categories':['00-damage', '01-whole'],
    'weights':'data1a-frozen10epochs.pth',
    'modelType':models.resnet50,
    'imageSize':299,
    'transforms':get_transforms()
}

model2 = {
    'name':'damageLocation',
    'categories':['00-front', '01-rear', '02-side'],
    'weights':'data2a-frozen10epochs-unfrozen10epochs.pth',
    'modelType':models.resnet50,
    'imageSize':299,
    'transforms':get_transforms()
}

model3 = {
    'name':'damageSeverity',
    'categories':['01-minor', '02-moderate', '03-severe'],
    'weights':'data3a-frozen10epochs-unfrozen10epochs.pth',
    'modelType':models.resnet50,
    'imageSize':299,
    'transforms':get_transforms()
}

qcrashModelDefs = [model1,model2,model3]

def generateLearner(modelDefition:dict, imagesPath:Path = Path("/tmp")):
    
    fnames = [
        "/{}_1.jpg".format(c)
        for c in modelDefition['categories']
        ]
    db=ImageDataBunch.from_name_re(
        imagesPath,
        fnames,
        r"/([^/]+)_\d+.jpg$",
        ds_tfms=modelDefition['transforms'],
        size=modelDefition['imageSize'],
    ).normalize(imagenet_stats)
    learner = create_cnn(db, modelDefition['modelType'])
    learner.model.load_state_dict(
        torch.load(modelDefition['weights'], map_location="cpu")
    )
    return learner

qcrashLearners = [generateLearner(md) for md in qcrashModelDefs]



""" #Model Categories
cat_fnames_l1 = [
    "/{}_1.jpg".format(c)
    for c in ['00-damage', '01-whole']
    ]

cat_fnames_l2 = [
    "/{}_1.jpg".format(c)
    for c in ['00-front', '01-rear', '02-side']
    ]

cat_fnames_l3 = [
    "/{}_1.jpg".format(c)
    for c in ['01-minor', '02-moderate', '03-severe']
    ]    


cat_data_l1 = ImageDataBunch.from_name_re(
    cat_images_path,
    cat_fnames_l1,
    r"/([^/]+)_\d+.jpg$",
    ds_tfms=get_transforms(),
    size=299,
).normalize(imagenet_stats)
cat_learner_l1 = create_cnn(cat_data_l1, models.resnet50)
cat_learner_l1.model.load_state_dict(
    torch.load("stage-1-50.pth", map_location="cpu")
) """


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = qcrashLearners[0].predict(img)
    return JSONResponse({
        "predicted_class": pred_class
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)


