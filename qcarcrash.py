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
import io
import uuid
import os
import time


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

""" model2 = {
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
 """
qcrashModelDefs = [model1]

tempPath = Path("/tmp")
#qcrashModelDefs = [model1]


def generateLearner(modelDefition:dict, imagesPath:Path = Path("/tmp")):
    
    modelDir=imagesPath / modelDefition['name']
    modelDir.mkdir

    fnames = [
        "/{}_1.jpg".format(c)
        for c in modelDefition['categories']
        ]
    print(fnames)
    db=ImageDataBunch.single_from_classes(
        modelDir,
        modelDefition['categories'],
        tfms=modelDefition['transforms'],
        size=modelDefition['imageSize']
    ).normalize(imagenet_stats)
    learner = create_cnn(db, modelDefition['modelType'])
    learner.model.load_state_dict(
        torch.load(modelDefition['weights'], map_location="cpu")
    )
    
    return learner





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
    
    tempFilePath = tempPath / (str(uuid.uuid1()) + '.jpg')
    tempFile = io.open(tempFilePath,'wb')
    tempFile.write(bytes)
    tempFile.close()

    img = open_image(tempFilePath)
    #img = open_image(BytesIO(bytes))
    
    pred,_,_=qcrashLearners[0].predict(img)
    pred_class_0 = [pred]
    pred,_,_=qcrashLearners[1].predict(img)
    pred_class_1 = [pred]
    pred,_,_=qcrashLearners[2].predict(img)
    pred_class_2 = [pred]

    os.remove(tempFilePath)

    return JSONResponse({
        qcrashModelDefs[0]['name']: pred_class_0,
        qcrashModelDefs[1]['name']: pred_class_1,
        qcrashModelDefs[2]['name']: pred_class_2
    })

def predict_image(path):
    
    

    img = open_image(path)
    #img = open_image(BytesIO(bytes))
    
    pred,_,_=qcrashLearners[0].predict(img)
    pred_class_0 = [pred]


    

    return JSONResponse({
        qcrashModelDefs[0]['name']: pred_class_0
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
        qcrashLearners = [generateLearner(md) for md in qcrashModelDefs]
        uvicorn.run(app, host="0.0.0.0", port=8008)
    elif "test" in sys.argv:
        qcrashLearners = [generateLearner(md) for md in qcrashModelDefs]
        print(predict_image(Path('./6b7c64a6-e1e6-11e8-a965-99eec267d82d.jpg')))



