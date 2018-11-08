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
import fastai.version
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

#This almost never works

model1 = {
    'name':'isDamaged',
    'categories':['damage', 'whole'],
    'weights':'data1a-frozen10epochs-paperspace-noz.pth',
    'modelType':models.resnet50,
    'imageSize':299,
    'transforms':get_transforms()
}

<<<<<<< HEAD
""" model2 = {
=======
#This almost always work

# model1 = {
#     'name':'isDamaged',
#     'categories':['bucatini_all_amatriciana', 'cappelletti_in_brodo', 'caprese', 'coda_alla_vaccinara', 'cotoletta_alla_milanese', 'lasagne', 'risotto_ai_frutti_di_mare', 'risotto_ai_funghi', 'risotto_alla_milanese', 'spaghetti_alla_carbonara', 'tagliatelle_al_ragu', 'tonnarelli_cacio_e_pepe', 'trenette_al_pesto'],
#     'weights':'stage-1-50.pth',
#     'modelType':models.resnet50,
#     'imageSize':299,
#     'transforms':get_transforms()
# }

model2 = {
>>>>>>> af33878e30973a7814330ac29fae6c339077be96
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
<<<<<<< HEAD
 """
qcrashModelDefs = [model1]
=======

#qcrashModelDefs = [model1,model2,model3]
>>>>>>> af33878e30973a7814330ac29fae6c339077be96

tempPath = Path("/tmp")
qcrashModelDefs = [model1]


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
<<<<<<< HEAD
        modelDefition['categories'],
        tfms=modelDefition['transforms'],
        size=modelDefition['imageSize']
=======
        fnames,
        r"/([^/]+)_\d+.jpg$",
        ds_tfms=modelDefition['transforms'],
        size=modelDefition['imageSize'],bs=32
>>>>>>> af33878e30973a7814330ac29fae6c339077be96
    ).normalize(imagenet_stats)
    learner = create_cnn(db, modelDefition['modelType'])
    learner.model.load_state_dict(
        torch.load(modelDefition['weights'], map_location="cpu")
    )
    
    return learner



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

    
    # pred,_,_=qcrashLearners[1].predict(img)
    # pred_class_1 = [pred]
    # pred,_,_=qcrashLearners[2].predict(img)
    # pred_class_2 = [pred]

    os.remove(tempFilePath)

    return JSONResponse({
        qcrashModelDefs[0]['name']: pred_class_0,
        # qcrashModelDefs[1]['name']: pred_class_1,
        # qcrashModelDefs[2]['name']: pred_class_2
    })

def predict_image(path):
    
    

    img = open_image(path)
    #img = open_image(BytesIO(bytes))
    
    pred,_,_=qcrashLearners[0].predict(img)
    #pred,_,_=cat_learner_l1.predict(img)
    pred_class_0 = [pred]
<<<<<<< HEAD

=======
    # pred,_,_=qcrashLearners[1].predict(img)
    # pred_class_1 = [pred]
    # pred,_,_=qcrashLearners[2].predict(img)
    # pred_class_2 = [pred]
>>>>>>> af33878e30973a7814330ac29fae6c339077be96

    

    return JSONResponse({
<<<<<<< HEAD
        qcrashModelDefs[0]['name']: pred_class_0
=======
        qcrashModelDefs[0]['name']: pred_class_0,
        # qcrashModelDefs[1]['name']: pred_class_1,
        # qcrashModelDefs[2]['name']: pred_class_2
>>>>>>> af33878e30973a7814330ac29fae6c339077be96
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
        for c in range(1,10):
            print(str(predict_image(Path('./6b7c64a6-e1e6-11e8-a965-99eec267d82d.jpg'))))
        print(fastai.__version__)




