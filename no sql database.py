from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

client = pymongo.MongoClient("mongodb+srv://ruelanannmargaret:<password>@diseasealert.4xrany4.mongodb.net/?retryWrites=true&w=majority&appName=diseasealert")
db = client["onion_bulb_diseases"]
collection = db["diseases"]

disease_data = [
    {
        "name": "Anthracnose-twister Disease",
        "causes": ["Fungal infection(Colletotrichum)"],
        "symptoms": ["white or pale-yellow water-soaked oval depressed lesions on leaf blades."],
        "prevention": ["Destruction of cull piles", "Apply fungicides", "Rotation out of onions for at least 2-3 years"],
        "treatment": ["No cure, remove infected bulbs"]
    },
    {
        "name": "Purple Blotch",
        "causes": ["Fungal infection(Alternaria porris)"],
        "symptoms": ["The lesions may girdle leaves/stalk and cause their drooping.", "The infected plants fail to develop bulbs", "Small, sunken, whitish flecks with purple coloured centres. on leaves and flower stalks"],
        "prevention": ["Crop rotation", "Use of protective fungicides"],
        "treatment": ["No cure, remove infected bulbs"]
    },
    {
        "name": "Downy Mildew",
        "causes": ["Fungal infection(Oomycete)"],
        "symptoms": ["On leaves, cottony white mycelial growth develops and appears white.", "small yellowish to orange flecks or streaks in the middle of the leaves, which soon develop into elongated spindle shaped spots surrounded by pinkish margin."],
        "prevention": ["Avoid warm and humid climate ", "Use of disease-free seed", "Apply fungicides"],
        "treatment": ["No cure, remove infected bulbs"]
    },
    {
        "name": "Stemphylium Leaf Blight",
        "causes": ["Fungal infection(Stemphylium vesicarium.)"],
        "symptoms": ["Sunken, collapsed tissues around the neck of the onion bulb", "Gray mold often occurs between the scales on the collapsed areas"],
        "prevention": ["Avoid excessive nitrogen", "Use crop rotation"],
        "treatment": ["Remove infected bulbs", "Maintain cool, dry storage conditions"]
    },
    {
        "name": "Iris Yellow Spot",
        "causes": ["Fungal infection"],
        "symptoms": ["Cream, elliptical spots on the leaves. Spots also appear on onion scapes or flower stalks of onions."],
        "prevention": ["Control thrips", "Control weeds and volunteer onion plants", "Remove and destroy infected plants", "Rotate crops"],
        "treatment": ["No cure, remove infected bulbs."]
    }
]

collection.insert_many(disease_data)

@app.route("/get_disease_info", methods=["POST"])
def get_disease_info_route():
    name = request.form["name"]
    query = {"name": name}
    result = collection.find_one(query)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Disease not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
