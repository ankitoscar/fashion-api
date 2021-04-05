from flask import Flask, jsonify, request
from image_recommend import ImageRecommender
import json

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to Fashion-API!'

@app.route('/v1/fashion/', methods=['GET'])
def get_links():
    link = request.args.get("image", None)
    imagerecommend = ImageRecommender()
    try:
        links = imagerecommend.get_similar_images(link)
        final_links = json.dumps(links)
        return final_links
    except Exception as e:
        error = 'You request couldn\tt be processed due to error: ' + str(e)
        return error

if __name__=='__main__':
    app.run(debug=True)