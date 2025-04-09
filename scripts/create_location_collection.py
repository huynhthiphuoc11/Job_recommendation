import pymongo

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017/"  # Thay bằng URI của bạn nếu cần
DATABASE_NAME = "Job-Recomendation"
COLLECTION_NAME = "all_locations_Data"

# Sample location data
locations = [
    {"location": "Ho Chi Minh City", "latitude": 10.8231, "longitude": 106.6297},
    {"location": "Hanoi", "latitude": 21.0285, "longitude": 105.8542},
    {"location": "Da Nang", "latitude": 16.0471, "longitude": 108.2068},
    {"location": "Can Tho", "latitude": 10.0452, "longitude": 105.7469},
    {"location": "Hai Phong", "latitude": 20.8449, "longitude": 106.6881},
]

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Insert location data
collection.insert_many(locations)

print(f"Inserted {len(locations)} locations into the '{COLLECTION_NAME}' collection.")