from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import SearchHistory, User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import pickle
import numpy as np

# popular_df = pickle.load(open('popular.pkl','rb'))
# pt = pickle.load(open('pt.pkl','rb'))
# books = pickle.load(open('books.pkl','rb'))
# similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

auth = Blueprint('auth', __name__)

@auth.route('/')
def index():
    return render_template('index.html'
                        #    book_name = list(popular_df['Book-Title'].values),
                        #    author=list(popular_df['Book-Author'].values),
                        #    image=list(popular_df['Image-URL-M'].values),
                        #    votes=list(popular_df['num_ratings'].values),
                        #    rating=list(popular_df['avg_rating'].values)
    )

from urllib.parse import unquote

# @auth.route('/book/<book_title>')
# @login_required
# def track_click(book_title):
#     book_title = unquote(book_title)

#     try:
#         index = np.where(pt.index == book_title)[0][0]
#         similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]

#         recommended_books = []
#         for i in similar_items:
#             temp_df = books[books['Book-Title'] == pt.index[i[0]]]
#             item = [
#                 temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
#                 temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
#                 temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
#             ]
#             recommended_books.append(item)
        
#         return render_template("dashboard.html", data=recommended_books, clicked_book=book_title)

#     except:
#         flash("Sorry, we couldn't generate recommendations for this book.", "error")
#         return redirect(url_for('auth.index'))


# @auth.route('/recommend')
# def recommend_ui():
#     return render_template('recommend.html')

@auth.route('/contact')
def contact():
    return render_template('contact.html')

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(
                email=email,
                first_name=first_name,
                password=generate_password_hash(password1, method="pbkdf2:sha256")
            )
            db.session.add(new_user)
            db.session.commit()

            flash('Account created!', category='success')
            return redirect(url_for('auth.index'))  # or redirect to homepage if you prefer
    return render_template('signup.html')



@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('auth.index'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)

@auth.route('/show_users')
def show_users():
    users = User.query.all()
    for user in users:
        print(user.email, user.first_name)
    return 'Check console for user info'


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/recommend_books', methods=['POST'])
@login_required
def recommend():
    user_input = request.form.get('user_input').strip()
    
    data = []
    no_results = False

    if user_input:
        # Find titles that contain the user input (case-insensitive)
        matched_titles = [title for title in pt.index if user_input.lower() in title.lower()]

        if matched_titles:
            # Save the first matched title to search history
            history = SearchHistory(user_id=current_user.id, search_term=matched_titles[0])
            db.session.add(history)
            db.session.commit()

            for title in matched_titles[:1]:  # Only use first match for similarity
                try:
                    index = np.where(pt.index == title)[0][0]
                    similar_items = sorted(
                        list(enumerate(similarity_scores[index])),
                        key=lambda x: x[1],
                        reverse=True
                    )[1:21]

                    for i in similar_items:
                        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                        item = [
                            temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
                            temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
                            temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
                        ]
                        data.append(item)
                except:
                    continue

            if not data:
                no_results = True

        else:
            # Fallback: search by author
            temp_df = books[books['Book-Author'].str.contains(user_input, case=False, na=False)]
            temp_df = temp_df.drop_duplicates('Book-Title').head(10)

            for _, row in temp_df.iterrows():
                item = [row['Book-Title'], row['Book-Author'], row['Image-URL-M']]
                data.append(item)

            if not data:
                no_results = True

    return render_template('recommend.html', data=data, no_results=no_results)



recommend_bp = Blueprint('dashboard', __name__)

@auth.route('/dashboard')
@login_required
def recommend_from_history():
    user_searches = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.id.desc()).limit(10).all()
    
    if not user_searches:
        return render_template('dashboard.html', no_data=True)

    recommended_books = []

    for search in user_searches:
        try:
            index = np.where(pt.index == search.search_term)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

            for i in similar_items:
                temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                item = [
                    temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
                    temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
                    temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
                ]
                recommended_books.append(item)
        except:
            continue

    # Remove duplicates
    seen = set()
    unique_recommendations = []
    for book in recommended_books:
        if book[0] not in seen:
            unique_recommendations.append(book)
            seen.add(book[0])

    return render_template('dashboard.html', data=unique_recommendations)




    

