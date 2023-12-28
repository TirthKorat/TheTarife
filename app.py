from flask import Flask, render_template, request
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from collections import defaultdict
from flask_paginate import Pagination

app = Flask(__name__)

# # Load dataset
# df = pd.read_csv('recipes.csv', usecols=['name', 'ingredients', 'steps', 'description', 'tags'])

df = pd.read_csv('recipes.csv', usecols=['name', 'ingredients', 'steps', 'description', 'tags'], low_memory=False)

# Replace missing values
df['tags'] = df['tags'].fillna('')
df['ingredients'] = df['ingredients'].fillna('')

# Combine tags and ingredients for searching
df['searchable_text'] = df['tags'] + ' ' + df['ingredients']

# Index for keyword searching
keyword_index = defaultdict(list)

for idx, row in df.iterrows():
    text = row['searchable_text']
    for keyword in text.split():
        keyword_index[keyword].append(row.to_dict())

# Load model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Pagination
PER_PAGE = 12

def get_recipes(offset, per_page, matching_recipes):
    return matching_recipes[offset: offset + per_page]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    search_input = request.form['search-input'].strip()  # Remove leading/trailing spaces
    keywords = search_input.split()

    # Use BERT to match keywords
    matching_recipes = []

    for keyword in keywords:
        # Check if the keyword is in the index
        if keyword in keyword_index:
            matching_recipes.extend(keyword_index[keyword])

    # Sort the recipes by relevance (number of matched keywords)
    if matching_recipes:
        matching_recipes = sorted(matching_recipes, key=lambda x: sum(1 for kw in keywords if kw in x['searchable_text']), reverse=True)

    # Pagination
    page = int(request.args.get('page', 1))
    total = len(matching_recipes)
    pagination = Pagination(page=page, total=total, record_name='recipes', per_page=PER_PAGE, bs_version=4)

    # Get the recipes for the current page
    recipes = get_recipes(pagination.page, PER_PAGE, matching_recipes)

    # Check if the request is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('recipes_partial.html', recipes=recipes, pagination=pagination)

    return render_template('index.html', recipes=recipes, pagination=pagination)


if __name__ == '__main__':
    app.run(debug=True)
