from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendation
import rec2

app = Flask(__name__)
CORS(app) 
        
@app.route('/cat', methods=['GET'])
def recommend_categories():
        res = recommendation.combine_results(request.args.get('title'))
        return jsonify(res)

@app.route('/dest', methods=['GET'])
def recommend_destinations():
        res = rec2.results(request.args.get('title'))
        return jsonify(res)

if __name__=='__main__':
        app.run(port = 5000, debug = True)
