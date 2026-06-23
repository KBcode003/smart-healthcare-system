import datetime
from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    search_term = db.Column(db.String(100))
class ClickLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    book_title = db.Column(db.String(255))
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    specialty = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    available = db.Column(db.Boolean, default=True)   




