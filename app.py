import random
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import os
import certifi
from pymongo import MongoClient
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField, IntegerField, TextAreaField
from wtforms.validators import DataRequired, Length, EqualTo
from wtforms.widgets import TextInput
from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALSModel
from flask.cli import load_dotenv
import logging
from bson import ObjectId
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize CSRF protection
csrf = CSRFProtect(app)

# MongoDB connection settings
mongo_uri = os.getenv('MONGO_URI')
mongo_db = "Hotel_Recommendation"

# Use certifi to get the path to the CA bundle
ca = certifi.where()

# Initialize the MongoClient with TLS settings
client = MongoClient(
    mongo_uri,
    tls=True,  # Enable TLS
    tlsCAFile=ca  # Path to CA bundle
)

# Access the database and collection
db = client[mongo_db]
users_collection = db["users"]
reviews_collection = db["review_with_index"]
offerings_collection = db["offerings"]

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class AutocompleteInput(TextInput):
    def __call__(self, field, **kwargs):
        kwargs.setdefault('data-provide', 'typeahead')
        kwargs.setdefault('autocomplete', 'off')
        return super().__call__(field, **kwargs)

class RatingForm(FlaskForm):
    hotel_name = StringField('Hotel Name', validators=[DataRequired()], widget=AutocompleteInput())
    hotel_id = StringField('Hotel ID', validators=[DataRequired()])
    rating = IntegerField('Rating', validators=[DataRequired()])
    review = TextAreaField('Review', validators=[Length(max=500)])
    submit = SubmitField('Submit Rating')

@app.route('/autocomplete_hotel', methods=['GET'])
def autocomplete_hotel():
    search = request.args.get('q')
    if search:
        hotels = offerings_collection.find({"name": {"$regex": search, "$options": "i"}})
        results = [{"id": str(hotel["_id"]), "name": hotel["name"], "address": hotel.get("address", "")} for hotel in hotels]
        return jsonify(matching_results=results)
    return jsonify(matching_results=[])


@app.route('/rate_hotel', methods=['GET', 'POST'])
@login_required
def rate_hotel():
    form = RatingForm()
    if form.validate_on_submit():
        hotel_id = form.hotel_id.data
        rating = form.rating.data
        review = form.review.data
        user_id = current_user.numeric_id

        # Insert the rating into the database
        reviews_collection.insert_one({
            "hotel_id": hotel_id,
            "numeric_id": user_id,
            "rating": rating,
            "review": review,
            "username": current_user.username  # Storing username for easier access later
        })

        flash('Your rating has been submitted!', 'success')
        return redirect(url_for('home'))
    
    return render_template('rate_hotel.html', form=form)

def generate_unique_numeric_id():
    while True:
        numeric_id = str(uuid.uuid4().int)[:12]  # Generates a 12-digit numeric ID
        if not (users_collection.find_one({"numeric_id": numeric_id}) or reviews_collection.find_one({"numeric_id": numeric_id})):
            return numeric_id

class User(UserMixin):
    def __init__(self, id, username, password_hash, numeric_id=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        user_data = reviews_collection.find_one({"username": username})
        if user_data:
            self.numeric_id = user_data["numeric_id"]
        else:
            self.numeric_id = generate_unique_numeric_id()

    @staticmethod
    def get(numeric_id):
        logging.info(f"Attempting to retrieve user with ID: {numeric_id}")
        user_data = users_collection.find_one({"numeric_id": numeric_id})
        if user_data:
            return User(str(user_data["_id"]), user_data["username"], user_data["password_hash"], user_data.get("numeric_id"))
        return None

    @staticmethod
    def create(username, password_hash):
        user_data = reviews_collection.find_one({"username": username})
        print("user data:::: ", user_data)
        if user_data:
            numeric_id = user_data["numeric_id"]
        else:
            numeric_id = generate_unique_numeric_id()
        user_id = users_collection.insert_one({
            "username": username,
            "password_hash": password_hash,
            "numeric_id": numeric_id
        }).inserted_id
        return User(str(user_id), username, password_hash, numeric_id)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()  
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if users_collection.find_one({"username": username}):
            flash('Username already exists')
            return redirect(url_for('signup'))

        new_user = User.create(username, hashed_password)
        login_user(new_user)
        flash('Your account has been created!', 'success')
        return redirect(url_for('home'))

    return render_template('signup.html', form=form)

@login_manager.user_loader
def load_user(user_id):
    logging.info(f'Loading user with ID: {user_id}')
    user = User.get(user_id)
    if user:
        logging.info(f'User {user.username} loaded successfully')
    else:
        logging.info(f'User with ID {user_id} not found')
    return user

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Recommendation System") \
    .getOrCreate()

# Load the ALSModel directly
recommender_model = ALSModel.load('models/als_model')

@app.route('/')
@login_required
def home():
    logging.info('Entered home route')
    logging.info(f'Session data: {session}')
    logging.info(f'Current user: {current_user}, Authenticated: {current_user.is_authenticated}')
    if current_user.is_authenticated:
        logging.info(f'Current user: {current_user.username}')
        recommendations = get_user_recommendations(current_user.username)
        logging.info(f'Recommendations for {current_user.username}: {recommendations}')
        return render_template('home.html', username=current_user.username, recommendations=recommendations)
    else:
        logging.info('No user is currently authenticated')
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user_data = users_collection.find_one({"username": username})

        if user_data and check_password_hash(user_data['password_hash'], password):
            ######
            user = User.get(user_data['numeric_id'])
            if not user.numeric_id:
                user.numeric_id = generate_unique_numeric_id()
                users_collection.update_one({"_id": user_data["_id"]}, {"$set": {"numeric_id": user.numeric_id}})
            login_user(user)
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))

        flash('Login unsuccessful. Please check username and password', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html', form=form)


def get_user_recommendations(username):
    user_reviews = reviews_collection.find_one({"username": username})
    if user_reviews:
        logging.info("found the user")
        user_id = user_reviews["numeric_id"]
        customer_df = spark.createDataFrame([Row(user_id=user_id)])
        logging.info("created a df")
        recommendations = recommender_model.recommendForUserSubset(customer_df, 10)
        logging.info("finished finding recommendations")
        recommendations_pd = recommendations.toPandas()
        if 'recommendations' in recommendations_pd.columns:
            recommended_items = recommendations_pd['recommendations'][0]
        else:
            recommended_items = []
    else:
        # If user does not have reviews, return top 10 recommended hotels
        recommendations = recommender_model.recommendForAllUsers(10).toPandas()
        recommended_items = []
        for rec in recommendations['recommendations']:
            recommended_items.extend(rec)
        recommended_items = recommended_items[:10]
    return recommended_items


@app.route('/logout')
@login_required
def logout():
    logging.info('Logging out user')
    logout_user()
    return redirect(url_for('home'))

@app.route('/recommend', methods=['GET'])
@login_required
def recommend():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'error': 'Customer ID is required'}), 400

    try:
        customer_id = int(customer_id)
    except ValueError:
        return jsonify({'error': 'Invalid Customer ID'}), 400

    # Create a Spark DataFrame for the customer
    customer_df = spark.createDataFrame([Row(user_id=customer_id)])

    # Generate top 5 restaurant recommendations for the customer
    recommendations = recommender_model.recommendForUserSubset(customer_df, 5)

    # Convert to Pandas DataFrame for easier manipulation
    recommendations_pd = recommendations.toPandas()

    # Extract recommended items
    if 'recommendations' in recommendations_pd.columns:
        recommended_items = recommendations_pd['recommendations'][0]
    else:
        recommended_items = []

    if len(recommended_items) == 0:
        return render_template('home.html', error='No recommendations found')

    return render_template('home.html', customer_id=customer_id, recommendations=recommended_items)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
