from flask import Flask, render_template, url_for, request, send_from_directory,jsonify,make_response,session,Response
import os
from AI_Vs_Human_Geoguess.utils.common import generate_image,actual_coords,model_predictions,paths,denormalize_lat,denormalize_lon,random_trash_talk,dist
from keras.models import load_model
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('page1.html', Title='Home')

@app.route('/aivshuman', methods=['GET', 'POST'])
def game():
    if request.method == 'POST' and 'generate' in request.form:
        result = generate_image() # Assuming this function returns just the filename
        
        response=make_response(render_template('AI.html', Title='Single Player', result=url_for('dataset_image', filename=result)))
        response.set_cookie('Image',result)
        response.set_cookie('page','AI')
        
        return response
    return render_template('AI.html', Title='AI VS You',result=None)

@app.route('/sologuess', methods=['GET', 'POST'])
def singleplayer():
    if request.method == 'POST' and 'generate' in request.form:
        result = generate_image()  # Assuming this function returns just the filename
        
        response = make_response(render_template('guessloc.html', Title='Single Player', result=url_for('dataset_image', filename=result)))
        response.set_cookie('Image', result)
        response.set_cookie('page', 'guessloc')
        return response
        


    # For GET request or if 'generate' button is not clicked
    return render_template('guessloc.html', Title='Single Player', result=None)

@app.route('/dataset/<path:filename>')
def dataset_image(filename):
    return send_from_directory(paths,filename)
@app.route('/save-coordinates', methods=['POST'])
def set_lan_lon():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lng = data.get('lng')
        
        if lat is None or lng is None:
            raise ValueError("Missing latitude or longitude")

        print(lat, lng)
        
        response = make_response(jsonify({'status': 'success'}))
        response.set_cookie('latitude', lat)
        response.set_cookie('longitude', lng)
        
        return response
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


        

@app.route('/map')
def map():
    image_url="static\images\Smily FAce.png"
    return render_template('map.html',image_url=image_url)
@app.route('/result',methods=['POST','GET'])
def result():
    try:
        lat = request.cookies.get('latitude')
        lng = request.cookies.get('longitude')
        page_name=request.cookies.get('page')
        Image_path=request.cookies.get('Image')
        print(Image_path)
        actual_lat,actual_lon=actual_coords(Image_path)
    except:
        return render_template(f'{page_name}.html')
    if page_name=='AI':
        trash_talk=None
        norm_model_lat,norm_model_lon=model_predictions(Image_path)
        model_lat,model_lon=denormalize_lat(norm_model_lat),denormalize_lon(norm_model_lon)
        human_distance=dist(lat1=actual_lat,lon1=actual_lon,lat2=float(lat),lon2=float(lng))
        ai_distance=dist(lat1=actual_lat,lon1=actual_lon,lat2=float(model_lat),lon2=float(model_lon))
        if human_distance<ai_distance:
            verdict=f'You Won You where mark was {ai_distance-human_distance} closer than AI'
            print(verdict)
            response=make_response(render_template('result2.html',actual_lat=actual_lat,actual_lon=actual_lon,lat=lat,lng=lng,model_prediction_lat=model_lat,model_prediction_lon=model_lon,verdict=verdict,trash_talk=None))
            response.delete_cookie('Image')
            response.delete_cookie('longitude')
            response.delete_cookie('latitude')
            return response
        else:
            verdict=f'You lost.Better Luck Next Time.AI was {human_distance-ai_distance} than You'
            trash_talk='🤖:'+random_trash_talk()
            print(verdict)
            print(trash_talk)
        
            response=make_response(render_template('result2.html',actual_lat=actual_lat,actual_lon=actual_lon,lat=lat,lng=lng,model_prediction_lat=model_lat,model_prediction_lon=model_lon,verdict=verdict,trash_talk=trash_talk))
            response.delete_cookie('Image')
            response.delete_cookie('longitude')
            response.delete_cookie('latitude')
            return response
    if page_name=='guessloc':
        model_lat,model_lon=0,0
        response=make_response(render_template('result2.html',actual_lat=actual_lat,actual_lon=actual_lon,lat=lat,lng=lng,model_prediction_lat=model_lat,model_prediction_lon=model_lon))
        response.delete_cookie('Image')
        response.delete_cookie('lng')
        response.delete_cookie('latitude')
        return response
    
    



    

if __name__ == '__main__':
    app.run(debug=True)
